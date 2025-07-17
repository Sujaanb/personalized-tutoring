
import React, { useState, useRef, useEffect } from 'react';
import { Send, Upload, FileText, MessageSquare, Trash2, Bot, User, Paperclip, History, Brain, BookOpen, Lightbulb, Target, TrendingUp, Zap } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import DashboardBackground from '../components/DashboardBackground';
import { useToast } from '@/hooks/use-toast';

interface ChatMessage {
  id: string;
  content: string;
  sender: 'user' | 'ai';
  timestamp: Date;
  type?: 'text' | 'document_analysis' | 'suggestion';
}

interface UploadedDocument {
  id: string;
  name: string;
  size: string;
  uploadTime: Date;
  type: string;
}

interface ChatSession {
  id: string;
  title: string;
  lastMessage: Date;
  messageCount: number;
  preview: string;
  messages: ChatMessage[];
}

export default function Dashboard() {
  const { toast } = useToast();
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: '1',
      content: "Welcome to your personalized AI tutor! I'm here to help you learn and grow. Upload documents, ask questions, or let me know what you'd like to study today. I can analyze your materials and provide tailored explanations based on your job role and learning goals.",
      sender: 'ai',
      timestamp: new Date(),
      type: 'text'
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [uploadedDocs, setUploadedDocs] = useState<UploadedDocument[]>([]);
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([
    {
      id: 'current',
      title: 'Current Session',
      lastMessage: new Date(),
      messageCount: 1,
      preview: 'Welcome to your personalized AI tutor...',
      messages: [{
        id: '1',
        content: "Welcome to your personalized AI tutor! I'm here to help you learn and grow. Upload documents, ask questions, or let me know what you'd like to study today. I can analyze your materials and provide tailored explanations based on your job role and learning goals.",
        sender: 'ai',
        timestamp: new Date(),
        type: 'text'
      }]
    }
  ]);
  const [currentSession, setCurrentSession] = useState('current');
  const [isLoading, setIsLoading] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputMessage.trim()) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      content: inputMessage,
      sender: 'user',
      timestamp: new Date(),
      type: 'text'
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    // Update current session with new message
    setChatSessions(prev => prev.map(session => 
      session.id === currentSession 
        ? { 
            ...session, 
            lastMessage: new Date(), 
            messageCount: session.messageCount + 1,
            messages: [...session.messages, userMessage]
          }
        : session
    ));

    // Simulate AI response with more variety
    setTimeout(() => {
      const responses = [
        `Great question! Based on your uploaded documents and our conversation, here's what I can tell you about "${inputMessage}". This relates to your professional development and I can break it down into key concepts that align with your role.`,
        `I understand you're exploring "${inputMessage}". Let me analyze this in the context of your career goals and provide some structured insights. Would you like me to create a learning path for this topic?`,
        `Excellent! "${inputMessage}" is an important topic. I can see from your knowledge base that this connects to several documents you've uploaded. Let me provide a comprehensive explanation tailored to your background.`,
        `That's a thoughtful question about "${inputMessage}". I'll draw from your uploaded materials and industry best practices to give you a detailed response that's relevant to your professional context.`
      ];

      const aiMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        content: responses[Math.floor(Math.random() * responses.length)],
        sender: 'ai',
        timestamp: new Date(),
        type: 'text'
      };
      setMessages(prev => [...prev, aiMessage]);
      setIsLoading(false);

      // Update session with AI response
      setChatSessions(prev => prev.map(session => 
        session.id === currentSession 
          ? { 
              ...session, 
              lastMessage: new Date(), 
              messageCount: session.messageCount + 1, 
              preview: aiMessage.content.substring(0, 50) + '...',
              messages: [...session.messages, aiMessage]
            }
          : session
      ));
    }, 1500);
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files) {
      Array.from(files).forEach(file => {
        const newDoc: UploadedDocument = {
          id: Date.now().toString() + Math.random(),
          name: file.name,
          size: (file.size / 1024).toFixed(1) + ' KB',
          uploadTime: new Date(),
          type: file.type || 'unknown'
        };
        setUploadedDocs(prev => [...prev, newDoc]);

        // Add document analysis message
        const analysisMessage: ChatMessage = {
          id: Date.now().toString() + '_analysis',
          content: `📄 Document uploaded: "${file.name}". I've analyzed the content and added it to your knowledge base. The insights from this document are now available for our conversation.`,
          sender: 'ai',
          timestamp: new Date(),
          type: 'document_analysis'
        };
        setMessages(prev => [...prev, analysisMessage]);
        
        // Update current session with document analysis
        setChatSessions(prev => prev.map(session => 
          session.id === currentSession 
            ? { 
                ...session, 
                lastMessage: new Date(), 
                messageCount: session.messageCount + 1,
                messages: [...session.messages, analysisMessage]
              }
            : session
        ));
      });
      
      toast({
        title: "Document processed successfully",
        description: "Your document has been analyzed and integrated into the knowledge base.",
      });
    }
  };

  const startNewSession = () => {
    const newSession: ChatSession = {
      id: Date.now().toString(),
      title: `Session ${chatSessions.length}`,
      lastMessage: new Date(),
      messageCount: 1,
      preview: 'New learning session started...',
      messages: [
        {
          id: '1',
          content: "New session started! How can I help you learn today?",
          sender: 'ai',
          timestamp: new Date(),
          type: 'text'
        }
      ]
    };
    
    setChatSessions(prev => [...prev, newSession]);
    setCurrentSession(newSession.id);
    setMessages(newSession.messages);
    setShowHistory(false);
  };

  const switchSession = (sessionId: string) => {
    const session = chatSessions.find(s => s.id === sessionId);
    if (session) {
      setCurrentSession(sessionId);
      setMessages(session.messages);
      setShowHistory(false);
      toast({
        title: "Session switched",
        description: `Loaded ${session.title} with ${session.messageCount} messages.`,
      });
    }
  };

  const clearCurrentSession = () => {
    const clearedMessage: ChatMessage = {
      id: '1',
      content: "Session cleared! How can I help you learn today?",
      sender: 'ai',
      timestamp: new Date(),
      type: 'text'
    };
    
    setMessages([clearedMessage]);
    setChatSessions(prev => prev.map(session => 
      session.id === currentSession 
        ? { 
            ...session, 
            messageCount: 1, 
            preview: 'Session cleared...', 
            lastMessage: new Date(),
            messages: [clearedMessage]
          }
        : session
    ));
    toast({
      title: "Session cleared",
      description: "All messages in current session have been removed.",
    });
  };

  const clearAllHistory = () => {
    const initialMessage: ChatMessage = {
      id: '1',
      content: "Welcome to your personalized AI tutor! I'm here to help you learn and grow. Upload documents, ask questions, or let me know what you'd like to study today.",
      sender: 'ai',
      timestamp: new Date(),
      type: 'text'
    };

    setChatSessions([{
      id: 'current',
      title: 'Current Session',
      lastMessage: new Date(),
      messageCount: 1,
      preview: 'Welcome to your personalized AI tutor...',
      messages: [initialMessage]
    }]);
    setCurrentSession('current');
    setMessages([initialMessage]);
    setShowHistory(false);
    
    toast({
      title: "All history cleared",
      description: "All chat sessions have been deleted.",
    });
  };

  const removeDocument = (docId: string) => {
    setUploadedDocs(prev => prev.filter(doc => doc.id !== docId));
    toast({
      title: "Document removed",
      description: "Document has been removed from the knowledge base.",
    });
  };

  const getFileIcon = (type: string) => {
    if (type.includes('pdf')) return '📄';
    if (type.includes('doc')) return '📝';
    if (type.includes('text')) return '📋';
    if (type.includes('image')) return '🖼️';
    return '📄';
  };

  return (
    <div className="min-h-screen relative overflow-hidden">
      <DashboardBackground />
      
      <div className="relative z-20 min-h-screen p-4">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-4 animate-fade-in">
            <div className="bg-gradient-to-r from-emerald-500 via-teal-600 to-cyan-600 p-3 rounded-xl shadow-lg animate-pulse-scale">
              <Brain className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-white animate-gradient-shift">AI Tutor Dashboard</h1>
              <p className="text-slate-300">Your personalized learning companion</p>
            </div>
          </div>
          <div className="flex items-center space-x-2 animate-fade-in">
            <Button
              onClick={() => setShowHistory(!showHistory)}
              variant="outline"
              className="bg-slate-800/50 border-slate-600 text-white hover:bg-slate-700/50 hover-scale"
            >
              <History className="w-4 h-4 mr-2" />
              History
            </Button>
            <Button
              onClick={startNewSession}
              variant="outline"
              className="bg-slate-800/50 border-slate-600 text-white hover:bg-slate-700/50 hover-scale"
            >
              <MessageSquare className="w-4 h-4 mr-2" />
              New Session
            </Button>
            <Button
              onClick={() => fileInputRef.current?.click()}
              variant="outline"
              className="bg-slate-800/50 border-slate-600 text-white hover:bg-slate-700/50 hover-scale"
            >
              <Upload className="w-4 h-4 mr-2" />
              Upload
            </Button>
          </div>
        </div>

        <div className="grid grid-cols-12 gap-6 h-[calc(100vh-140px)]">
          {/* Sidebar */}
          <div className="col-span-3 space-y-4">
            {/* Learning Stats */}
            <Card className="bg-slate-800/30 backdrop-blur-xl border-slate-700/50 animate-scale-in">
              <CardHeader className="pb-3">
                <CardTitle className="text-white text-lg flex items-center">
                  <TrendingUp className="w-5 h-5 mr-2 animate-float-rotate" />
                  Learning Progress
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center justify-between animate-fade-in">
                  <div className="flex items-center space-x-2">
                    <BookOpen className="w-4 h-4 text-emerald-400 animate-symbol-float" />
                    <span className="text-sm text-slate-300">Documents</span>
                  </div>
                  <Badge variant="secondary" className="bg-emerald-500/20 text-emerald-400 animate-pulse-scale">
                    {uploadedDocs.length}
                  </Badge>
                </div>
                <div className="flex items-center justify-between animate-fade-in">
                  <div className="flex items-center space-x-2">
                    <MessageSquare className="w-4 h-4 text-cyan-400 animate-symbol-float" />
                    <span className="text-sm text-slate-300">Messages</span>
                  </div>
                  <Badge variant="secondary" className="bg-cyan-500/20 text-cyan-400 animate-pulse-scale">
                    {messages.length}
                  </Badge>
                </div>
                <div className="flex items-center justify-between animate-fade-in">
                  <div className="flex items-center space-x-2">
                    <TrendingUp className="w-4 h-4 text-teal-400 animate-symbol-float" />
                    <span className="text-sm text-slate-300">Sessions</span>
                  </div>
                  <Badge variant="secondary" className="bg-teal-500/20 text-teal-400 animate-pulse-scale">
                    {chatSessions.length}
                  </Badge>
                </div>
              </CardContent>
            </Card>

            {/* Chat History */}
            {showHistory && (
              <Card className="bg-slate-800/30 backdrop-blur-xl border-slate-700/50 animate-slide-in-right">
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-white text-lg flex items-center">
                      <History className="w-5 h-5 mr-2 animate-float-rotate" />
                      Chat History
                    </CardTitle>
                    <Button
                      onClick={clearAllHistory}
                      variant="ghost"
                      size="sm"
                      className="text-red-400 hover:text-red-300 hover:bg-red-500/10"
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </div>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-48">
                    <div className="space-y-2">
                      {chatSessions.map((session, index) => (
                        <div
                          key={session.id}
                          onClick={() => switchSession(session.id)}
                          className={`p-3 rounded-lg cursor-pointer transition-all duration-200 hover-scale animate-fade-in ${
                            session.id === currentSession
                              ? 'bg-emerald-500/20 border border-emerald-500/30'
                              : 'bg-slate-900/50 hover:bg-slate-800/50'
                          }`}
                          style={{ animationDelay: `${index * 0.1}s` }}
                        >
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-sm font-medium text-white">{session.title}</span>
                            <span className="text-xs text-slate-400">{session.messageCount}</span>
                          </div>
                          <p className="text-xs text-slate-400 truncate">{session.preview}</p>
                          <p className="text-xs text-slate-500 mt-1">
                            {session.lastMessage.toLocaleDateString()}
                          </p>
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
            )}

            {/* Knowledge Base */}
            <Card className="bg-slate-800/30 backdrop-blur-xl border-slate-700/50 flex-1 animate-scale-in">
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center text-white">
                  <FileText className="w-5 h-5 mr-2 animate-float-rotate" />
                  Knowledge Base
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-[300px]">
                  <div className="space-y-2">
                    {uploadedDocs.map((doc, index) => (
                      <div
                        key={doc.id}
                        className="flex items-center justify-between p-3 bg-slate-900/50 rounded-lg border border-slate-700/30 hover-scale animate-fade-in"
                        style={{ animationDelay: `${index * 0.1}s` }}
                      >
                        <div className="flex items-center space-x-3 flex-1">
                          <span className="text-lg animate-symbol-float">{getFileIcon(doc.type)}</span>
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium text-white truncate">{doc.name}</p>
                            <p className="text-xs text-slate-400">{doc.size}</p>
                          </div>
                        </div>
                        <Button
                          onClick={() => removeDocument(doc.id)}
                          variant="ghost"
                          size="sm"
                          className="h-8 w-8 p-0 text-slate-400 hover:text-red-400 hover-scale"
                        >
                          <Trash2 className="w-3 h-3" />
                        </Button>
                      </div>
                    ))}
                    {uploadedDocs.length === 0 && (
                      <div className="text-center py-8 text-slate-400 animate-fade-in">
                        <div className="w-16 h-16 mx-auto mb-4 bg-slate-700/30 rounded-full flex items-center justify-center animate-pulse-scale">
                          <FileText className="w-8 h-8 opacity-50 animate-symbol-float" />
                        </div>
                        <p className="mb-2">No documents uploaded yet</p>
                        <p className="text-xs">Upload documents to enhance AI responses</p>
                      </div>
                    )}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
          </div>

          {/* Main Chat Panel */}
          <div className="col-span-9">
            <Card className="bg-slate-800/30 backdrop-blur-xl border-slate-700/50 h-full flex flex-col animate-scale-in">
              <CardHeader className="border-b border-slate-700/30">
                <CardTitle className="flex items-center justify-between text-white">
                  <div className="flex items-center space-x-2">
                    <div className="w-3 h-3 bg-emerald-400 rounded-full animate-pulse"></div>
                    <span>Interactive Learning Session</span>
                  </div>
                  <div className="flex items-center space-x-4">
                    <span className="text-sm text-slate-400 flex items-center animate-fade-in">
                      <Zap className="w-4 h-4 mr-1 animate-symbol-float" />
                      AI Powered
                    </span>
                    <Button
                      onClick={clearCurrentSession}
                      variant="ghost"
                      size="sm"
                      className="text-slate-400 hover:text-red-400 hover-scale"
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </div>
                </CardTitle>
              </CardHeader>
              <CardContent className="flex-1 flex flex-col p-6">
                {/* Messages */}
                <ScrollArea className="flex-1 mb-6 pr-4">
                  <div className="space-y-6">
                    {messages.map((message, index) => (
                      <div
                        key={message.id}
                        className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'} animate-fade-in`}
                        style={{ animationDelay: `${index * 0.1}s` }}
                      >
                        <div
                          className={`max-w-[80%] p-4 rounded-2xl hover-scale ${
                            message.sender === 'user'
                              ? 'bg-gradient-to-r from-cyan-600 to-blue-600 text-white'
                              : message.type === 'document_analysis'
                              ? 'bg-gradient-to-r from-emerald-600/20 to-teal-600/20 text-emerald-100 border border-emerald-500/30'
                              : 'bg-slate-900/50 text-slate-100 border border-slate-700/30'
                          }`}
                        >
                          <div className="flex items-start space-x-3">
                            {message.sender === 'ai' && (
                              <div className="bg-gradient-to-r from-emerald-500 to-cyan-500 p-1.5 rounded-lg animate-pulse-scale">
                                <Bot className="w-4 h-4 text-white" />
                              </div>
                            )}
                            {message.sender === 'user' && (
                              <div className="bg-white/20 p-1.5 rounded-lg animate-pulse-scale">
                                <User className="w-4 h-4 text-white" />
                              </div>
                            )}
                            <div className="flex-1">
                              <p className="text-sm leading-relaxed">{message.content}</p>
                              <div className="flex items-center justify-between mt-2">
                                <p className="text-xs opacity-70">
                                  {message.timestamp.toLocaleTimeString()}
                                </p>
                                {message.type === 'document_analysis' && (
                                  <Badge variant="outline" className="text-xs border-emerald-500/30 text-emerald-400 animate-pulse-scale">
                                    Document Analysis
                                  </Badge>
                                )}
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                    {isLoading && (
                      <div className="flex justify-start animate-fade-in">
                        <div className="bg-slate-900/50 p-4 rounded-2xl border border-slate-700/30">
                          <div className="flex items-center space-x-3">
                            <div className="bg-gradient-to-r from-emerald-500 to-cyan-500 p-1.5 rounded-lg animate-pulse-scale">
                              <Bot className="w-4 h-4 text-white" />
                            </div>
                            <div className="flex space-x-1">
                              <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce"></div>
                              <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                              <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                    <div ref={messagesEndRef} />
                  </div>
                </ScrollArea>

                {/* Input Area */}
                <div className="bg-slate-900/30 rounded-xl p-4 border border-slate-700/30 animate-fade-in">
                  <div className="flex items-end space-x-3">
                    <div className="flex-1 relative">
                      <Textarea
                        value={inputMessage}
                        onChange={(e) => setInputMessage(e.target.value)}
                        placeholder="Ask me anything about your documents, request explanations, or start a new learning topic..."
                        className="min-h-[80px] bg-slate-800/50 border-slate-600 text-white resize-none pr-12 placeholder:text-slate-400"
                        onKeyPress={(e) => {
                          if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            handleSendMessage();
                          }
                        }}
                      />
                      <Button
                        onClick={() => fileInputRef.current?.click()}
                        variant="ghost"
                        size="sm"
                        className="absolute right-2 bottom-2 h-8 w-8 p-0 text-slate-400 hover:text-cyan-400 hover-scale"
                      >
                        <Paperclip className="w-4 h-4" />
                      </Button>
                    </div>
                    <Button
                      onClick={handleSendMessage}
                      disabled={!inputMessage.trim() || isLoading}
                      className="h-[80px] bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-700 hover:to-blue-700 text-white px-6 shadow-lg hover-scale"
                    >
                      <Send className="w-5 h-5" />
                    </Button>
                  </div>
                  <div className="flex items-center justify-between mt-3 text-xs text-slate-400">
                    <span>Press Enter to send, Shift+Enter for new line</span>
                    <div className="flex items-center space-x-4">
                      <span className="flex items-center animate-fade-in">
                        <Target className="w-3 h-3 mr-1 animate-symbol-float" />
                        {uploadedDocs.length} docs in context
                      </span>
                      <span className="flex items-center animate-fade-in">
                        <Lightbulb className="w-3 h-3 mr-1 animate-symbol-float" />
                        AI-powered responses
                      </span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>

      {/* Hidden File Input */}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".pdf,.doc,.docx,.txt,.md,.json,.csv"
        onChange={handleFileUpload}
        className="hidden"
      />
    </div>
  );
}
