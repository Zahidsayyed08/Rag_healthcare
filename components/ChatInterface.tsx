"use client"

import { useState } from "react"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Loader2, AlertTriangle } from "lucide-react"
import ReactMarkdown from "react-markdown"
import { Alert, AlertDescription } from "@/components/ui/alert"

interface Medicine {
  id?: string
  name: string
  genericName?: string
  description: string
  indications: string
  warnings: string
  dosage: string
}

interface Message {
  role: "user" | "assistant"
  content: string
}

interface ChatInterfaceProps {
  medicine: Medicine
}

export default function ChatInterface({ medicine }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [error, setError] = useState<string>("")
  const [isLoading, setIsLoading] = useState(false)
  const isLLMData = medicine.id?.startsWith("gemini-")

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const userMessage = input.trim()
    setInput("")
    setError("")
    setIsLoading(true)

    const userMessageObj: Message = { role: "user", content: userMessage }
    const newMessages: Message[] = [...messages, userMessageObj]
    setMessages(newMessages)

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: newMessages,
          medicineData: medicine,
        }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || "Failed to get response")
      }

      const assistantMessage: Message = {
        role: "assistant",
        content: data.response,
      }

      setMessages([...newMessages, assistantMessage])
    } catch (err) {
      console.error("Chat error:", err)
      setError("Failed to send message. Please try again.")
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="mt-4 space-y-4">
      <h3 className="text-lg font-semibold">Chat about {medicine.name}</h3>

      {isLLMData && (
        <Alert>
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            This information is based on medical research. For the most accurate information, please consult official
            sources or healthcare providers.
          </AlertDescription>
        </Alert>
      )}

      <div className="h-96 overflow-y-auto border rounded-lg p-4 space-y-4 bg-background">
        {messages.map((message, index) => (
          <div key={index} className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}>
            <div
              className={`max-w-[80%] rounded-lg px-4 py-2 ${
                message.role === "user"
                  ? "bg-primary text-primary-foreground"
                  : "bg-secondary text-secondary-foreground"
              }`}
            >
              <ReactMarkdown className="prose dark:prose-invert max-w-none">{message.content}</ReactMarkdown>
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex justify-center">
            <Loader2 className="h-6 w-6 animate-spin" />
          </div>
        )}

        {error && <div className="text-destructive text-center">{error}</div>}
      </div>

      <form onSubmit={handleSubmit} className="flex gap-2">
        <Input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about this medicine..."
          className="flex-grow text-stone-500"
          disabled={isLoading}
        />
        <Button type="submit" disabled={isLoading || !input.trim()}>
          {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : "Send"}
        </Button>
      </form>
    </div>
  )
}

