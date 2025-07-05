package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"strings"

	markdown "github.com/MichaelMure/go-term-markdown"
	"github.com/ollama/ollama/api"
)

const (
	ollamaAPIURL = "http://localhost:11434/api/chat"
	historyFile  = "owllama_chat_history.json"
)

func printUsage() {
	fmt.Println("Usage: owllama <command> [args]")
	fmt.Println("Commands:")
	fmt.Println("  list")
	fmt.Println("  generate <model> <prompt>")
	fmt.Println("  version")
	fmt.Println("  help")
	fmt.Println("  chat <model>")
	fmt.Println("  history <list|view> [key]")
}

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create Ollama client: %v\n", err)
		os.Exit(1)
	}

	ctx := context.Background()
	cmd := os.Args[1]

	// Command dispatcher for easier extensibility
	commands := map[string]func(context.Context, *api.Client){
		"list":     handleList,
		"generate": handleGenerate,
		"version":  handleVersion,
		"help":     func(_ context.Context, _ *api.Client) { printUsage() },
		"chat":     handleChat,
		"history":  handleHistory,
	}

	if handler, ok := commands[cmd]; ok {
		handler(ctx, client)
		return
	}

	// Forward unknown commands to ollama executable
	forwardToOllama(os.Args[1:])
}

func handleList(ctx context.Context, client *api.Client) {
	resp, err := client.List(ctx)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error listing models: %v\n", err)
		os.Exit(1)
	}
	for _, m := range resp.Models {
		fmt.Println(m.Name)
	}
}

func handleGenerate(ctx context.Context, client *api.Client) {
	if len(os.Args) < 4 {
		fmt.Println("Usage: owllama generate <model> <prompt>")
		os.Exit(1)
	}
	model := os.Args[2]
	prompt := os.Args[3]
	req := &api.GenerateRequest{
		Model:  model,
		Prompt: prompt,
		Stream: nil, // disables streaming, returns full response
	}
	err := client.Generate(ctx, req, func(resp api.GenerateResponse) error {
		fmt.Print(resp.Response)
		if resp.Done {
			fmt.Println()
		}
		return nil
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error generating response: %v\n", err)
		os.Exit(1)
	}
}

func handleVersion(ctx context.Context, client *api.Client) {
	ver, err := client.Version(ctx)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error getting version: %v\n", err)
		os.Exit(1)
	}
	fmt.Println(ver)
}

func handleChat(_ context.Context, _ *api.Client) {
	reader := bufio.NewReader(os.Stdin)
	model := "gemma3"
	if len(os.Args) >= 3 {
		model = os.Args[2]
	}
	fullPrompt, err := buildPrompt(reader)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Prompt error: %v\n", err)
		return
	}

	sessionKey := generateSessionKey()
	chatHistory := loadChatHistory()
	session := ChatSession{
		SessionKey: sessionKey,
		Messages:   []ChatMessage{},
	}
	var messages []map[string]string
	messages = append(messages, map[string]string{"role": "user", "content": fullPrompt})

	responseText, err := ollamaChatAPI(model, messages)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error contacting Ollama API: %v\n", err)
	} else {
		printText := filterQwenThink(model, responseText)
		fmt.Println("\033[1;36mUser:\033[0m")
		printMarkdown(fullPrompt)
		fmt.Println("\033[1;32mOllama:\033[0m")
		printMarkdown(printText)
		messages = append(messages, map[string]string{"role": "assistant", "content": responseText})
		session.Messages = append(session.Messages, ChatMessage{Role: "user", Content: fullPrompt})
		session.Messages = append(session.Messages, ChatMessage{Role: "ollama", Content: responseText})
	}
	fmt.Println("\nType /exit to quit. Type /clear to reset context.")

	for {
		fmt.Print("\033[1;36mYou: \033[0m")
		prompt, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				fmt.Println("\nExiting chat session.")
				break
			}
			fmt.Println("Error reading input:", err)
			continue
		}
		prompt = strings.TrimSpace(prompt)

		if prompt == "/edit" || prompt == "/vi" {
			editor := os.Getenv("EDITOR")
			if editor == "" {
				editor = "vi"
			}
			tmpfile, err := os.CreateTemp("", "owllama_edit_*.md")
			if err != nil {
				fmt.Fprintf(os.Stderr, "Could not create temp file: %v\n", err)
				continue
			}
			tmpfile.Close()
			cmd := exec.Command(editor, tmpfile.Name())
			cmd.Stdin = os.Stdin
			cmd.Stdout = os.Stdout
			cmd.Stderr = os.Stderr
			if err := cmd.Run(); err != nil {
				fmt.Fprintf(os.Stderr, "Editor error: %v\n", err)
				os.Remove(tmpfile.Name())
				continue
			}
			content, err := os.ReadFile(tmpfile.Name())
			os.Remove(tmpfile.Name())
			if err != nil {
				fmt.Fprintf(os.Stderr, "Could not read temp file: %v\n", err)
				continue
			}
			prompt = strings.TrimSpace(string(content))
			if prompt == "" {
				continue
			}
		}

		if strings.HasPrefix(prompt, "/search ") {
			query := strings.TrimSpace(strings.TrimPrefix(prompt, "/search "))
			if query == "" {
				fmt.Println("Please provide a search query.")
				continue
			}
			fmt.Println("Searching the internet for:", query)
			result, err := searchInternet(query)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Search error: %v\n", err)
				continue
			}
			fmt.Println("Search result:", result)
			messages = append(messages, map[string]string{"role": "assistant", "content": result})
			continue
		}
		if prompt == "/exit" {
			fmt.Println("Exiting chat session.")
			break
		}
		if prompt == "/clear" {
			messages = nil
			fmt.Println("Context cleared.")
			continue
		}
		if prompt == "" {
			continue
		}
		messages = append(messages, map[string]string{"role": "user", "content": prompt})
		responseText, err := ollamaChatAPI(model, messages)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error contacting Ollama API: %v\n", err)
			continue
		}
		printText := filterQwenThink(model, responseText)
		fmt.Println("\033[1;36mYou:\033[0m")
		printMarkdown(prompt)
		fmt.Println("\033[1;32mOllama:\033[0m")
		printMarkdown(printText)
		messages = append(messages, map[string]string{"role": "assistant", "content": responseText})
		session.Messages = append(session.Messages, ChatMessage{Role: "user", Content: prompt})
		session.Messages = append(session.Messages, ChatMessage{Role: "ollama", Content: responseText})
	}
	userInputFound := false
	for _, msg := range session.Messages {
		if msg.Role == "user" && strings.TrimSpace(msg.Content) != "" {
			userInputFound = true
			break
		}
	}
	if userInputFound {
		chatHistory.Sessions = append(chatHistory.Sessions, session)
		saveChatHistory(chatHistory)
	}
}

func printMarkdown(md string) {
	out := markdown.Render(md, 80, 6)
	os.Stdout.Write(out)
}

func handleHistory(_ context.Context, _ *api.Client) {
	if len(os.Args) < 3 {
		fmt.Println("Usage: owllama history <list|view> [key]")
		os.Exit(1)
	}
	subcmd := os.Args[2]
	switch subcmd {
	case "list":
		chatHistory := loadChatHistory()
		for _, session := range chatHistory.Sessions {
			firstPrompt := ""
			for _, msg := range session.Messages {
				if msg.Role == "user" {
					firstPrompt = msg.Content
					break
				}
			}
			if len(firstPrompt) > 40 {
				firstPrompt = firstPrompt[:40]
			}
			fmt.Printf("%s: %s\n", session.SessionKey, firstPrompt)
		}
	case "view":
		if len(os.Args) < 4 {
			fmt.Println("Usage: owllama history view <key>")
			os.Exit(1)
		}
		key := os.Args[3]
		chatHistory := loadChatHistory()
		for _, session := range chatHistory.Sessions {
			if session.SessionKey == key {
				for _, msg := range session.Messages {
					role := msg.Role
					if len(role) > 0 {
						role = strings.ToUpper(role[:1]) + role[1:]
					}
					fmt.Printf("%s: %s\n", role, msg.Content)
				}
				return
			}
		}
		fmt.Println("Session not found.")
	default:
		fmt.Println("Usage: owllama history <list|view> [key]")
		os.Exit(1)
	}
}

func buildPrompt(reader *bufio.Reader) (string, error) {
	fmt.Println("\nWelcome to Owllama Chat!\nLet's build your first prompt step by step for best results.")
	fmt.Println("Step 1: Who should the AI act as? (Role/Persona)")
	fmt.Println("Example: Expert Chef")
	fmt.Print("Enter a role/persona: ")
	role, _ := reader.ReadString('\n')
	role = strings.TrimSpace(role)
	fmt.Println("\nStep 2: What do you want the AI to do? (Task)")
	fmt.Println("Example: Suggest a three-course vegetarian meal")
	fmt.Print("Enter a task: ")
	task, _ := reader.ReadString('\n')
	task = strings.TrimSpace(task)
	fmt.Println("\nStep 3: Any relevant context, constraints, or details?")
	fmt.Println("Example: Considering a Mediterranean diet, with a focus on fresh herbs and olive oil")
	fmt.Print("Enter context/details: ")
	contextDetails, _ := reader.ReadString('\n')
	contextDetails = strings.TrimSpace(contextDetails)
	fmt.Println("\nStep 4: How should the answer be presented? (Format/Output Request)")
	fmt.Println("Example: Provide the recipes in a clear, step-by-step format with estimated prep and cook times.")
	fmt.Print("Enter format/output request: ")
	formatRequest, _ := reader.ReadString('\n')
	formatRequest = strings.TrimSpace(formatRequest)
	fullPrompt := role + " - " + task + " - " + contextDetails + " - " + formatRequest
	fmt.Println("\nYour full prompt:")
	fmt.Println(fullPrompt)
	fmt.Print("\nPress Enter to continue or type 'edit' to start over: ")
	confirm, _ := reader.ReadString('\n')
	confirm = strings.TrimSpace(confirm)
	if strings.ToLower(confirm) == "edit" {
		fmt.Println("Restarting prompt setup...")
		return buildPrompt(reader)
	}
	return fullPrompt, nil
}

func ollamaChatAPI(model string, messages []map[string]string) (string, error) {
	bodyMap := struct {
		Model    string              `json:"model"`
		Messages []map[string]string `json:"messages"`
	}{
		Model:    model,
		Messages: messages,
	}
	bodyBytes, _ := json.Marshal(bodyMap)
	req, _ := http.NewRequest("POST", ollamaAPIURL, bytes.NewBuffer(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	respBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		return "", fmt.Errorf("ollama API error: %s", string(respBody))
	}
	var responseText string
	respScanner := bufio.NewScanner(strings.NewReader(string(respBody)))
	for respScanner.Scan() {
		line := respScanner.Text()
		var apiResp struct {
			Message struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"message"`
			Done bool `json:"done"`
		}
		if err := json.Unmarshal([]byte(line), &apiResp); err != nil {
			continue
		}
		if apiResp.Message.Role == "assistant" {
			responseText += apiResp.Message.Content
		}
		if apiResp.Done {
			break
		}
	}
	return responseText, nil
}

func filterQwenThink(model, text string) string {
	if !strings.Contains(model, "qwen3") {
		return text
	}
	for {
		start := strings.Index(text, "<think>")
		end := strings.Index(text, "</think>")
		if start != -1 && end != -1 && end > start {
			text = text[:start] + text[end+7:]
		} else {
			break
		}
	}
	return text
}

func generateSessionKey() string {
	b := make([]byte, 4)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

func loadChatHistory() ChatHistory {
	var chatHistory ChatHistory
	if f, err := os.Open(historyFile); err == nil {
		defer f.Close()
		json.NewDecoder(f).Decode(&chatHistory)
	}
	return chatHistory
}

func saveChatHistory(chatHistory ChatHistory) {
	f, ferr := os.OpenFile(historyFile, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	if ferr == nil {
		enc := json.NewEncoder(f)
		enc.SetIndent("", "  ")
		enc.Encode(chatHistory)
		f.Close()
	} else {
		fmt.Fprintf(os.Stderr, "Error writing to history file: %v\n", ferr)
	}
}

func forwardToOllama(args []string) {
	ollamaPath, err := execLookPath("ollama")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Could not find ollama executable in PATH.\n")
		os.Exit(1)
	}
	cmd := execCommand(ollamaPath, args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	err = cmd.Run()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error running ollama: %v\n", err)
		os.Exit(1)
	}
}

// execLookPath and execCommand are wrappers for testability
var execLookPath = func(file string) (string, error) {
	return exec.LookPath(file)
}
var execCommand = func(name string, arg ...string) *exec.Cmd {
	return exec.Command(name, arg...)
}

// ChatMessage represents a single message in a chat session
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatSession represents a chat session
type ChatSession struct {
	SessionKey string        `json:"session_key"`
	Messages   []ChatMessage `json:"messages"`
}

// ChatHistory is a collection of chat sessions
type ChatHistory struct {
	Sessions []ChatSession `json:"sessions"`
}

// searchInternet performs a simple web search using Wikipedia's API and returns a summary of the top result.
func searchInternet(query string) (string, error) {
	// Use Wikipedia's summary API
	apiURL := "https://en.wikipedia.org/api/rest_v1/page/summary/" + urlQueryEscapeWiki(query)
	resp, err := http.Get(apiURL)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode == 404 {
		return "No relevant information found.", nil
	}
	if resp.StatusCode != 200 {
		return "", fmt.Errorf("wikipedia API error: %s", resp.Status)
	}
	var data struct {
		Extract string `json:"extract"`
		Title   string `json:"title"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return "", err
	}
	if data.Extract != "" {
		return fmt.Sprintf("%s: %s", data.Title, data.Extract), nil
	}
	return "No relevant information found.", nil
}

// urlQueryEscapeWiki escapes a string for use in a Wikipedia API URL.
func urlQueryEscapeWiki(s string) string {
	s = strings.ReplaceAll(s, " ", "_")
	s = strings.ReplaceAll(s, "\n", "")
	s = strings.ReplaceAll(s, "\r", "")
	return s
}
