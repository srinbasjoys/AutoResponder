import React, { useState, useRef, useEffect } from 'react';
import { Mic, MicOff, Settings, MessageCircle, Brain, Zap, Trash2, ChevronDown } from 'lucide-react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [conversations, setConversations] = useState([]);
  const [currentProvider, setCurrentProvider] = useState('groq');
  const [currentModel, setCurrentModel] = useState('llama-3.1-8b-instant');
  const [providers, setProviders] = useState([]);
  const [availableModels, setAvailableModels] = useState({});
  const [showSettings, setShowSettings] = useState(false);
  const [sessionId] = useState(() => `session_${Date.now()}`);
  const [recordingTime, setRecordingTime] = useState(0);
  const [transcribedText, setTranscribedText] = useState('');
  
  const mediaRecorderRef = useRef(null);
  const recordingTimerRef = useRef(null);
  const streamRef = useRef(null);

  useEffect(() => {
    fetchAvailableModels();
    fetchProviders();
    fetchConversations();
    // Set up Groq as default with API key
    setupDefaultProvider();
  }, []);

  const setupDefaultProvider = async () => {
    try {
      // Set up Groq with the provided API key for testing
      await axios.post(`${BACKEND_URL}/api/providers`, {
        name: 'groq',
        api_key: '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',
        model: 'mixtral-8x7b-32768'
      });
      console.log('Default Groq provider set up successfully');
    } catch (error) {
      console.error('Error setting up default provider:', error);
    }
  };

  const fetchAvailableModels = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/api/models`);
      setAvailableModels(response.data.models);
    } catch (error) {
      console.error('Error fetching available models:', error);
    }
  };

  const fetchProviders = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/api/providers`);
      setProviders(response.data.providers);
    } catch (error) {
      console.error('Error fetching providers:', error);
    }
  };

  const fetchConversations = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/api/conversations/${sessionId}`);
      setConversations(response.data.conversations);
    } catch (error) {
      console.error('Error fetching conversations:', error);
    }
  };

  const clearConversations = async () => {
    try {
      await axios.delete(`${BACKEND_URL}/api/conversations/${sessionId}`);
      setConversations([]);
    } catch (error) {
      console.error('Error clearing conversations:', error);
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      
      const audioChunks = [];
      mediaRecorder.ondataavailable = (event) => {
        audioChunks.push(event.data);
      };
      
      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks);
        const reader = new FileReader();
        reader.onloadend = async () => {
          const base64Audio = reader.result.split(',')[1];
          await processAudio(base64Audio);
        };
        reader.readAsDataURL(audioBlob);
        
        // Stop all tracks
        stream.getTracks().forEach(track => track.stop());
      };
      
      mediaRecorder.start();
      setIsRecording(true);
      setRecordingTime(0);
      setTranscribedText('');
      
      // Start timer
      recordingTimerRef.current = setInterval(() => {
        setRecordingTime(prev => {
          if (prev >= 30) {
            stopRecording();
            return prev;
          }
          return prev + 1;
        });
      }, 1000);
      
    } catch (error) {
      console.error('Error starting recording:', error);
      alert('Error accessing microphone. Please check permissions.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      clearInterval(recordingTimerRef.current);
      setRecordingTime(0);
    }
  };

  const processAudio = async (audioData) => {
    setIsProcessing(true);
    try {
      const response = await axios.post(`${BACKEND_URL}/api/process-audio`, {
        audio_data: audioData,
        session_id: sessionId,
        provider: currentProvider,
        model: currentModel
      });
      
      // Set transcribed text
      setTranscribedText(response.data.user_input);
      
      // Add new conversation to the list
      setConversations(prev => [...prev, response.data]);
      
    } catch (error) {
      console.error('Error processing audio:', error);
      alert('Error processing audio. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleProviderSave = async (providerData) => {
    try {
      await axios.post(`${BACKEND_URL}/api/providers`, providerData);
      fetchProviders();
      setShowSettings(false);
    } catch (error) {
      console.error('Error saving provider:', error);
      alert('Error saving provider configuration.');
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50 to-indigo-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-6xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-blue-500 rounded-lg">
                <Brain className="w-6 h-6 text-white" />
              </div>
              <h1 className="text-2xl font-bold text-gray-900">AutoResponder</h1>
              <span className="px-2 py-1 bg-green-100 text-green-800 text-xs font-medium rounded-full">
                AI Assistant
              </span>
            </div>
            <div className="flex items-center space-x-2">
              <button
                onClick={clearConversations}
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
                title="Clear Conversation"
              >
                <Trash2 className="w-5 h-5 text-gray-600" />
              </button>
              <button
                onClick={() => setShowSettings(!showSettings)}
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <Settings className="w-5 h-5 text-gray-600" />
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-6xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Main Recording Interface */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-2xl shadow-lg p-8">
              <div className="text-center mb-8">
                <h2 className="text-3xl font-bold text-gray-900 mb-2">
                  Voice Assistant
                </h2>
                <p className="text-gray-600">
                  Hold the button to record up to 30 seconds of audio
                </p>
              </div>

              {/* Model Selection */}
              <div className="mb-8">
                <div className="flex flex-wrap gap-4 justify-center">
                  <div className="flex items-center space-x-2">
                    <label className="text-sm font-medium text-gray-700">Provider:</label>
                    <select
                      value={currentProvider}
                      onChange={(e) => {
                        setCurrentProvider(e.target.value);
                        // Reset to first available model for new provider
                        const models = availableModels[e.target.value] || [];
                        if (models.length > 0) {
                          setCurrentModel(models[0]);
                        }
                      }}
                      className="px-3 py-1 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500"
                    >
                      {Object.keys(availableModels).map(provider => (
                        <option key={provider} value={provider} className="capitalize">
                          {provider}
                        </option>
                      ))}
                    </select>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <label className="text-sm font-medium text-gray-700">Model:</label>
                    <select
                      value={currentModel}
                      onChange={(e) => setCurrentModel(e.target.value)}
                      className="px-3 py-1 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500"
                    >
                      {(availableModels[currentProvider] || []).map(model => (
                        <option key={model} value={model}>
                          {model}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>

              {/* Recording Interface */}
              <div className="flex flex-col items-center space-y-6">
                
                {/* Recording Button */}
                <div className="relative">
                  <button
                    onMouseDown={startRecording}
                    onMouseUp={stopRecording}
                    onTouchStart={startRecording}
                    onTouchEnd={stopRecording}
                    disabled={isProcessing}
                    className={`
                      w-32 h-32 rounded-full flex items-center justify-center text-white font-semibold text-lg
                      transition-all duration-200 transform hover:scale-105 active:scale-95
                      ${isRecording 
                        ? 'bg-red-500 recording-animation shadow-lg shadow-red-500/30' 
                        : 'bg-blue-500 hover:bg-blue-600 shadow-lg shadow-blue-500/30'
                      }
                      ${isProcessing ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
                    `}
                  >
                    {isProcessing ? (
                      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white"></div>
                    ) : isRecording ? (
                      <MicOff className="w-8 h-8" />
                    ) : (
                      <Mic className="w-8 h-8" />
                    )}
                  </button>
                  
                  {/* Recording Timer */}
                  {isRecording && (
                    <div className="absolute -bottom-8 left-1/2 transform -translate-x-1/2">
                      <div className="bg-red-500 text-white px-3 py-1 rounded-full text-sm font-medium">
                        {recordingTime}s / 30s
                      </div>
                    </div>
                  )}
                </div>

                {/* Status Text */}
                <div className="text-center min-h-12">
                  {isProcessing && (
                    <div className="flex items-center justify-center space-x-2 text-blue-600">
                      <Zap className="w-4 h-4 animate-pulse" />
                      <span className="font-medium">Processing audio...</span>
                    </div>
                  )}
                  {isRecording && (
                    <div className="flex items-center justify-center space-x-2 text-red-600">
                      <div className="audio-visualizer">
                        <div className="audio-bar"></div>
                        <div className="audio-bar"></div>
                        <div className="audio-bar"></div>
                        <div className="audio-bar"></div>
                        <div className="audio-bar"></div>
                      </div>
                      <span className="font-medium ml-2">Recording...</span>
                    </div>
                  )}
                  {!isRecording && !isProcessing && (
                    <p className="text-gray-500">
                      Press and hold to start recording
                    </p>
                  )}
                  
                  {/* Show transcribed text */}
                  {transcribedText && !isProcessing && (
                    <div className="mt-2 p-3 bg-blue-50 rounded-lg">
                      <p className="text-sm text-blue-800">
                        <strong>You said:</strong> "{transcribedText}"
                      </p>
                    </div>
                  )}
                </div>

                {/* Current Provider */}
                <div className="flex items-center space-x-2 bg-gray-50 px-4 py-2 rounded-lg">
                  <Brain className="w-4 h-4 text-gray-600" />
                  <span className="text-sm text-gray-600">Current AI:</span>
                  <span className="text-sm font-medium text-gray-900 capitalize">
                    {currentProvider} / {currentModel}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Conversation History */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-2xl shadow-lg p-6">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center space-x-2">
                  <MessageCircle className="w-5 h-5 text-blue-500" />
                  <h3 className="text-xl font-semibold text-gray-900">
                    Conversation
                  </h3>
                </div>
                <span className="text-xs text-gray-500">Last 5 interactions</span>
              </div>
              
              <div className="space-y-4 max-h-96 overflow-y-auto">
                {conversations.length === 0 ? (
                  <div className="text-center py-8">
                    <MessageCircle className="w-12 h-12 text-gray-300 mx-auto mb-3" />
                    <p className="text-gray-500 text-sm">
                      No conversations yet. Start by recording your first message!
                    </p>
                  </div>
                ) : (
                  conversations.map((conv, index) => (
                    <div key={conv.id || index} className="space-y-2">
                      <div className="bg-blue-50 p-3 rounded-lg">
                        <p className="text-sm text-gray-700">
                          <strong>You:</strong> {conv.user_input}
                        </p>
                        <p className="text-xs text-gray-500 mt-1">
                          {conv.provider} / {conv.model}
                        </p>
                      </div>
                      <div className="bg-gray-50 p-3 rounded-lg">
                        <p className="text-sm text-gray-700">
                          <strong>AI:</strong> {conv.ai_response}
                        </p>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Settings Panel */}
        {showSettings && (
          <SettingsPanel
            providers={providers}
            availableModels={availableModels}
            currentProvider={currentProvider}
            currentModel={currentModel}
            onProviderChange={setCurrentProvider}
            onModelChange={setCurrentModel}
            onProviderSave={handleProviderSave}
            onClose={() => setShowSettings(false)}
          />
        )}
      </div>
    </div>
  );
}

// Settings Panel Component
function SettingsPanel({ 
  providers, 
  availableModels,
  currentProvider, 
  currentModel,
  onProviderChange, 
  onModelChange,
  onProviderSave, 
  onClose 
}) {
  const [selectedProvider, setSelectedProvider] = useState(currentProvider);
  const [selectedModel, setSelectedModel] = useState(currentModel);
  const [apiKey, setApiKey] = useState('');

  const handleSave = () => {
    if (!apiKey.trim()) {
      alert('Please enter an API key');
      return;
    }

    onProviderSave({
      name: selectedProvider,
      api_key: apiKey,
      model: selectedModel
    });

    onProviderChange(selectedProvider);
    onModelChange(selectedModel);
    setApiKey('');
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-2xl p-8 max-w-lg w-full mx-4 max-h-90vh overflow-y-auto">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-2xl font-bold text-gray-900">AI Provider Settings</h3>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 text-2xl"
          >
            ×
          </button>
        </div>

        <div className="space-y-6">
          {/* Provider Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select AI Provider
            </label>
            <select
              value={selectedProvider}
              onChange={(e) => {
                setSelectedProvider(e.target.value);
                // Reset to first available model for new provider
                const models = availableModels[e.target.value] || [];
                if (models.length > 0) {
                  setSelectedModel(models[0]);
                }
              }}
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              {Object.keys(availableModels).map(provider => (
                <option key={provider} value={provider} className="capitalize">
                  {provider}
                </option>
              ))}
            </select>
          </div>

          {/* Model Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select Model
            </label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              {(availableModels[selectedProvider] || []).map(model => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
            </select>
          </div>

          {/* API Key */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              API Key
            </label>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="Enter your API key"
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
            <p className="text-xs text-gray-500 mt-1">
              Your API key will be securely stored and used for AI responses.
            </p>
          </div>

          {/* Save Button */}
          <button
            onClick={handleSave}
            className="w-full bg-blue-500 hover:bg-blue-600 text-white font-semibold py-3 px-4 rounded-lg transition-colors"
          >
            Save Configuration
          </button>

          {/* Current Providers Status */}
          <div className="border-t pt-4">
            <h4 className="text-sm font-medium text-gray-700 mb-3">Configured Providers:</h4>
            <div className="space-y-2">
              {providers.map(provider => (
                <div key={provider.name} className="flex items-center justify-between text-sm p-2 bg-gray-50 rounded">
                  <div>
                    <span className="capitalize font-medium">{provider.name}</span>
                    {provider.model && (
                      <span className="text-gray-500 ml-2">({provider.model})</span>
                    )}
                  </div>
                  <span className={`px-2 py-1 rounded text-xs ${
                    provider.configured 
                      ? 'bg-green-100 text-green-800' 
                      : 'bg-red-100 text-red-800'
                  }`}>
                    {provider.configured ? 'Configured' : 'Not Configured'}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;