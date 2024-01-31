'use client';

import { useState, useEffect, useRef } from 'react'

import Head from 'next/head'
import Header from '../components/Header'
import ChatMessages from '../components/ChatMessages'
import InputBar from '../components/InputBar'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';

export default function Home() {
    const [messages, setMessages] = useState([])
    const [collectedCodeBlocks, setCollectedCodeBlocks] = useState([]);
    const [input, setInput] = useState('')
    const inputRef = useRef(null)
    const [expandedBlocks, setExpandedBlocks] = useState(new Set());

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            handleSubmit(e)
        }
    }

    const handleCollectCodeBlock = (code) => {
        setCollectedCodeBlocks((prevBlocks) => [...prevBlocks, code]);
    };

    const clearMessages = () => {
        setMessages([]);
        setCollectedCodeBlocks([]); 
    }

    const importMessages = (importedMessages) => {
        setMessages(importedMessages);
      };

    const toggleCodeBlock = (index) => {
        setExpandedBlocks((prevExpanded) => {
          const updatedExpanded = new Set(prevExpanded);
          if (prevExpanded.has(index)) {
            updatedExpanded.delete(index);
          } else {
            updatedExpanded.add(index);
          }
          return updatedExpanded;
        });
      };

    const getSystemMessage = async (userInputMessage) => {
        const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/system_message`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${process.env.NEXT_PUBLIC_TTR_API_KEY}`
            },
            body: JSON.stringify(userInputMessage),
        })
        const systemMessage = await response.json();
        return { text: systemMessage.system_message, sender: 'systemMessage' }
    }

    const handleCommitCodeSnippets = async () => {
        try {
          const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/create_commit`, {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${process.env.NEXT_PUBLIC_TTR_API_KEY}`
            },
            body: JSON.stringify(collectedCodeBlocks.map(block => ({ diff: block }))),
          });

          if (!response.ok) {
            throw new Error('Error creating commit');
          }

          const data = await response.json();
          if (data.status === 'success') {
            console.log('Commit created successfully');
          } else {
            console.error('Error creating commit');
          }
        } catch (error) {
          console.error('Error creating commit:', error);
        }
      };

    const handleSubmit = async (e) => {
        e.preventDefault()

        let updatedMessages = []
        if (input.trim()) {
            const userInputMessage = { text: input, sender: 'user' }
            if (messages.length === 0) {
                const systemMessage = await getSystemMessage(userInputMessage);
                updatedMessages = [systemMessage, userInputMessage];
            } else {
                updatedMessages = [...messages, userInputMessage]
            }
            setMessages(updatedMessages)

            await handleChat(updatedMessages)

            setInput('')
        }
    }

    useEffect(() => {
        if (inputRef.current) {
            inputRef.current.style.height = 'auto'
            inputRef.current.style.height = inputRef.current.scrollHeight + 'px'
        }
    }, [input])

    const handleChat = async (updatedMessages) => {
        let accumulatedText = "";
        fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/chat_stream`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${process.env.NEXT_PUBLIC_TTR_API_KEY}`
            },
            body: JSON.stringify(updatedMessages),
        })
        .then(response => {
            const reader = response.body.getReader();
            return new ReadableStream({
                async start(controller) {
                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) {
                            break;
                        }
                        let newToken = new TextDecoder().decode(value);
                        accumulatedText += newToken;
                        controller.enqueue(newToken);
                    }
                    controller.close();
                    reader.releaseLock();
                }
            });
        })
        .then(stream => {
            updatedMessages = [...updatedMessages, { text: '', sender: 'llm' }];
            setMessages(updatedMessages);
            const reader = stream.getReader();
            reader.read().then(function processText({ done, value }) {
                if (done) {
                    return;
                }
                setMessages((prevMessages) => {
                    let outputMessage = prevMessages[prevMessages.length - 1];
                    outputMessage.text = accumulatedText;
                    return [...prevMessages.slice(0, -1), outputMessage];
                });
                return reader.read().then(processText);
            });
        });
    };

    return (<>
        <Head>
            <title>Talk to Repo</title>
            <meta
                name="description"
                content="Load any GitHub repository in a chat app."
            />
            <link rel="icon" href="/favicon.ico" />
            <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet" />
        </Head>

        <div className="h-screen flex flex-col bg-gray-800 text-gray-100 font-sans font-roboto">
            <Header 
                importMessages={importMessages} 
                clearMessages={clearMessages} 
                messages={messages}  
                handleCommitCodeSnippets={handleCommitCodeSnippets}
            />
            <div className="flex-1 overflow-auto p-4">
                <div className="flex flex-wrap md:flex-nowrap justify-center md:space-x-4">
                    <div className="w-full md:w-3/4 xl:w-3/5 md:max-w-screen-md order-last md:order-none">
                        <div className="flex justify-between h-full">
                        <div
                            className={`w-full ${
                                collectedCodeBlocks.length > 0 ? "lg:w-3/5" : "lg:w-full"
                            } overflow-auto`}
                        >
                                <ChatMessages messages={messages} onCollectCodeBlock={handleCollectCodeBlock} />
                            </div>
                            {collectedCodeBlocks.length > 0 && (
                                <div className="w-full lg:w-2/5 lg:ml-4 overflow-auto">
                                    {collectedCodeBlocks.map((code, index) => (
                                        <div
                                            key={index}
                                            className="whitespace-pre-wrap bg-gray-100 text-gray-800
                                                max-w-md text-xs p-2 rounded-lg shadow-md cursor-pointer my-2"
                                            onClick={() => toggleCodeBlock(index)}
                                        >
                                            <SyntaxHighlighter
                                                style={oneDark}
                                                customStyle={{
                                                    backgroundColor: "#2d2d2d",
                                                    borderRadius: "0.375rem",
                                                    padding: "1rem",
                                                }}
                                            >
                                                {code}
                                            </SyntaxHighlighter>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>

            <div className="border-t border-gray-700">
                <InputBar
                    input={input}
                    setInput={setInput}
                    handleKeyDown={handleKeyDown}
                    handleSubmit={handleSubmit}
                />
            </div>
        </div>     
    </>);    
}