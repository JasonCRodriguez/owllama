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
	"strings"
	"time"

	"github.com/ollama/ollama/api"
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
	switch cmd {
	case "list":
		resp, err := client.List(ctx)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error listing models: %v\n", err)
			os.Exit(1)
		}
		for _, m := range resp.Models {
			fmt.Println(m.Name)
		}
	case "generate":
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
	case "version":
		ver, err := client.Version(ctx)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error getting version: %v\n", err)
			os.Exit(1)
		}
		fmt.Println(ver)
	case "help":
		printUsage()
	case "chat":
		if len(os.Args) < 3 {
			fmt.Println("Usage: owllama chat <model>")
			os.Exit(1)
		}
		model := os.Args[2]
		historyFile := "owllama_chat_history.json"
		fmt.Println("Starting chat session. Type /exit to quit. Type /clear to reset context.")
		reader := bufio.NewReader(os.Stdin)
		// Generate a random session key for this chat
		sessionKeyBytes := make([]byte, 8)
		_, _ = rand.Read(sessionKeyBytes)
		sessionKey := hex.EncodeToString(sessionKeyBytes)

		// Load existing chat history
		var chatHistory ChatHistory
		if f, err := os.Open(historyFile); err == nil {
			defer f.Close()
			json.NewDecoder(f).Decode(&chatHistory)
		}

		// Create a new session
		session := ChatSession{
			SessionKey: sessionKey,
			Messages:   []ChatMessage{},
		}

		// Chat context for this session
		var messages []map[string]string // for HTTP API: role/content

		for {
			fmt.Print("You: ")
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

			// Add user message to context
			messages = append(messages, map[string]string{"role": "user", "content": prompt})

			// Show a spinner while processing
			spinnerDone := make(chan struct{})
			go func() {
				spinnerChars := []rune{'|', '/', '-', '\\'}
				idx := 0
				for {
					select {
					case <-spinnerDone:
						fmt.Print("\r           \r") // clear spinner
						return
					default:
						fmt.Printf("\rProcessing... %c", spinnerChars[idx%len(spinnerChars)])
						idx++
					}
					time.Sleep(100 * time.Millisecond)
				}
			}()

			// Prepare HTTP request to Ollama API
			ollamaURL := "http://localhost:11434/api/chat"
			bodyMap := map[string]interface{}{
				"model":    model,
				"messages": messages,
			}
			bodyBytes, _ := json.Marshal(bodyMap)
			req, _ := http.NewRequest("POST", ollamaURL, bytes.NewBuffer(bodyBytes))
			req.Header.Set("Content-Type", "application/json")
			resp, err := http.DefaultClient.Do(req)
			close(spinnerDone)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error contacting Ollama API: %v\n", err)
				continue
			}
			respBody, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			if resp.StatusCode != 200 {
				fmt.Fprintf(os.Stderr, "Ollama API error: %s\n", string(respBody))
				continue
			}
			// Parse response
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
					continue // skip malformed lines
				}
				if apiResp.Message.Role == "assistant" {
					responseText += apiResp.Message.Content
				}
				if apiResp.Done {
					break
				}
			}
			printText := responseText
			if strings.Contains(model, "qwen3") {
				for {
					start := strings.Index(printText, "<think>")
					end := strings.Index(printText, "</think>")
					if start != -1 && end != -1 && end > start {
						printText = printText[:start] + printText[end+7:]
					} else {
						break
					}
				}
			}
			fmt.Println("Ollama:", strings.TrimSpace(printText))

			// Add model response to context
			messages = append(messages, map[string]string{"role": "assistant", "content": responseText})

			// Append messages to session for history
			session.Messages = append(session.Messages, ChatMessage{Role: "user", Content: prompt})
			session.Messages = append(session.Messages, ChatMessage{Role: "ollama", Content: responseText})
		}
		// Save session to history only if user provided input
		userInputFound := false
		for _, msg := range session.Messages {
			if msg.Role == "user" && strings.TrimSpace(msg.Content) != "" {
				userInputFound = true
				break
			}
		}
		if userInputFound {
			chatHistory.Sessions = append(chatHistory.Sessions, session)
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
	case "history":
		if len(os.Args) < 3 {
			fmt.Println("Usage: owllama history <list|view> [key]")
			os.Exit(1)
		}
		subcmd := os.Args[2]
		historyFile := "owllama_chat_history.json"
		switch subcmd {
		case "list":
			var chatHistory ChatHistory
			file, err := os.Open(historyFile)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error opening history file: %v\n", err)
				os.Exit(1)
			}
			defer file.Close()
			if err := json.NewDecoder(file).Decode(&chatHistory); err != nil {
				fmt.Fprintf(os.Stderr, "Error parsing history file: %v\n", err)
				os.Exit(1)
			}
			for _, session := range chatHistory.Sessions {
				firstUserMsg := ""
				for _, msg := range session.Messages {
					if msg.Role == "user" {
						firstUserMsg = msg.Content
						break
					}
				}
				if len(firstUserMsg) > 40 {
					firstUserMsg = firstUserMsg[:40]
				}
				fmt.Printf("%s: %s\n", session.SessionKey, firstUserMsg)
			}
		case "view":
			if len(os.Args) < 4 {
				fmt.Println("Usage: owllama history view <key>")
				os.Exit(1)
			}
			key := os.Args[3]
			var chatHistory ChatHistory
			file, err := os.Open(historyFile)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error opening history file: %v\n", err)
				os.Exit(1)
			}
			defer file.Close()
			if err := json.NewDecoder(file).Decode(&chatHistory); err != nil {
				fmt.Fprintf(os.Stderr, "Error parsing history file: %v\n", err)
				os.Exit(1)
			}
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
		return
	}
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
