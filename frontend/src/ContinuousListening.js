import React, { useState, useEffect, useRef } from 'react';
import { Mic, MicOff, Zap, Volume2, VolumeX, Brain, Activity, Pause, Play } from 'lucide-react';

const ContinuousListening = ({ 
  currentProvider, 
  currentModel, 
  onTranscription, 
  onAIResponse,
  isVisible = true 
}) => {
  // State management
  const [isListening, setIsListening] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isResponding, setIsResponding] = useState(false);
  const [currentTranscription, setCurrentTranscription] = useState('');
  const [partialTranscription, setPartialTranscription] = useState('');
  const [aiResponse, setAiResponse] = useState('');
  const [voiceActivity, setVoiceActivity] = useState(false);
  const [sessionId, setSessionId] = useState('');
  const [chunkIndex, setChunkIndex] = useState(0);
  const [error, setError] = useState('');
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  
  // Audio recording state
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const [audioChunks, setAudioChunks] = useState([]);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  
  // Refs
  const streamRef = useRef(null);
  const pollingRef = useRef(null);
  const recordingTimerRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const voiceActivityRef = useRef(null);
  
  // Backend URL
  const backendUrl = process.env.REACT_APP_BACKEND_URL || 'https://b8dbe72f-8d81-45fb-a90a-90537128a55e.preview.emergentagent.com';
  
  // Voice activity detection
  const detectVoiceActivity = () => {
    if (!analyserRef.current) return false;
    
    const bufferLength = analyserRef.current.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    analyserRef.current.getByteFrequencyData(dataArray);
    
    // Calculate average volume
    const average = dataArray.reduce((sum, value) => sum + value, 0) / bufferLength;
    const threshold = 30; // Adjust based on environment
    
    return average > threshold;
  };
  
  // Start voice activity monitoring
  const startVoiceActivityMonitoring = () => {
    if (voiceActivityRef.current) {
      cancelAnimationFrame(voiceActivityRef.current);
    }
    
    const monitor = () => {
      if (audioContextRef.current && analyserRef.current) {
        const hasVoiceActivity = detectVoiceActivity();
        setVoiceActivity(hasVoiceActivity);
        
        if (isListening) {
          voiceActivityRef.current = requestAnimationFrame(monitor);
        }
      }
    };
    
    monitor();
  };
  
  // Initialize audio context
  const initializeAudio = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 16000
        } 
      });
      
      streamRef.current = stream;
      
      // Set up audio context for voice activity detection
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      const source = audioContextRef.current.createMediaStreamSource(stream);
      analyserRef.current = audioContextRef.current.createAnalyser();
      analyserRef.current.fftSize = 2048;
      source.connect(analyserRef.current);
      
      // Set up media recorder
      const recorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });
      
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          setAudioChunks(prev => [...prev, event.data]);
        }
      };
      
      recorder.onstop = () => {
        processAudioChunks();
      };
      
      setMediaRecorder(recorder);
      return true;
      
    } catch (error) {
      console.error('Error initializing audio:', error);
      setError('Failed to access microphone. Please check permissions.');
      return false;
    }
  };
  
  // Process audio chunks
  const processAudioChunks = async () => {
    if (audioChunks.length === 0) return;
    
    try {
      // Combine audio chunks
      const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
      
      // Convert to base64
      const reader = new FileReader();
      reader.onload = async () => {
        const base64Audio = reader.result.split(',')[1];
        
        // Send to backend
        await sendAudioChunk(base64Audio, false);
        
        // Clear chunks
        setAudioChunks([]);
        setChunkIndex(prev => prev + 1);
      };
      
      reader.readAsDataURL(audioBlob);
      
    } catch (error) {
      console.error('Error processing audio chunks:', error);
      setError('Failed to process audio');
    }
  };
  
  // Send audio chunk to backend
  const sendAudioChunk = async (audioData, isFinal = false) => {
    try {
      const response = await fetch(`${backendUrl}/api/continuous-audio`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          provider: currentProvider,
          model: currentModel,
          audio_chunk: audioData,
          chunk_index: chunkIndex,
          is_final: isFinal,
          voice_activity_detected: voiceActivity,
          noise_reduction: true,
          noise_reduction_strength: 0.7,
          auto_gain_control: true,
          high_pass_filter: true
        })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      
      const data = await response.json();
      console.log('Audio chunk sent:', data);
      
    } catch (error) {
      console.error('Error sending audio chunk:', error);
      setError('Failed to send audio data');
    }
  };
  
  // Start continuous listening
  const startListening = async () => {
    try {
      setError('');
      setConnectionStatus('connecting');
      
      // Initialize audio if not already done
      if (!mediaRecorder) {
        const success = await initializeAudio();
        if (!success) return;
      }
      
      // Generate session ID
      const newSessionId = `session_${Date.now()}`;
      setSessionId(newSessionId);
      
      // Start listening session
      const response = await fetch(`${backendUrl}/api/start-listening`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: newSessionId,
          provider: currentProvider,
          model: currentModel,
          continuous_mode: true,
          max_duration: 300
        })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      
      const data = await response.json();
      console.log('Started listening:', data);
      
      // Start recording
      if (mediaRecorder && mediaRecorder.state === 'inactive') {
        setIsListening(true);
        setIsRecording(true);
        setConnectionStatus('connected');
        
        // Start recording with 2-second intervals
        mediaRecorder.start(2000);
        
        // Start voice activity monitoring
        startVoiceActivityMonitoring();
        
        // Start polling for updates
        startPolling(newSessionId);
        
        // Start recording timer
        setRecordingTime(0);
        recordingTimerRef.current = setInterval(() => {
          setRecordingTime(prev => prev + 1);
        }, 1000);
      }
      
    } catch (error) {
      console.error('Error starting listening:', error);
      setError('Failed to start listening');
      setConnectionStatus('error');
    }
  };
  
  // Stop continuous listening
  const stopListening = async () => {
    try {
      setIsListening(false);
      setIsRecording(false);
      setConnectionStatus('disconnected');
      
      // Stop recording
      if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
      }
      
      // Stop voice activity monitoring
      if (voiceActivityRef.current) {
        cancelAnimationFrame(voiceActivityRef.current);
      }
      
      // Stop polling
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
      }
      
      // Stop recording timer
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current);
      }
      
      // Send stop request to backend
      if (sessionId) {
        const response = await fetch(`${backendUrl}/api/stop-listening`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            session_id: sessionId,
            reason: 'user_stopped'
          })
        });
        
        if (response.ok) {
          const data = await response.json();
          console.log('Stopped listening:', data);
        }
      }
      
      // Reset state
      setCurrentTranscription('');
      setPartialTranscription('');
      setAiResponse('');
      setVoiceActivity(false);
      setChunkIndex(0);
      setRecordingTime(0);
      
    } catch (error) {
      console.error('Error stopping listening:', error);
      setError('Failed to stop listening');
    }
  };
  
  // Start polling for conversation updates
  const startPolling = (sessionId) => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
    }
    
    pollingRef.current = setInterval(async () => {
      try {
        const response = await fetch(`${backendUrl}/api/conversation-state/${sessionId}`);
        
        if (response.ok) {
          const data = await response.json();
          
          setIsProcessing(data.is_processing);
          setIsResponding(data.is_responding);
          setCurrentTranscription(data.current_transcription);
          setPartialTranscription(data.partial_transcription);
          setAiResponse(data.ai_response);
          setVoiceActivity(data.voice_activity_detected);
          
          // Call callbacks
          if (data.current_transcription && onTranscription) {
            onTranscription(data.current_transcription);
          }
          
          if (data.ai_response && onAIResponse) {
            onAIResponse(data.ai_response);
          }
        }
        
      } catch (error) {
        console.error('Error polling conversation state:', error);
      }
    }, 500); // Poll every 500ms
  };
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
      }
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current);
      }
      if (voiceActivityRef.current) {
        cancelAnimationFrame(voiceActivityRef.current);
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, []);
  
  // Format recording time
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };
  
  if (!isVisible) return null;
  
  return (
    <div className="bg-white rounded-2xl shadow-lg p-8 mb-8">
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold text-gray-900 mb-2">
          Continuous Listening
        </h2>
        <p className="text-gray-600">
          Real-time conversation with voice activity detection
        </p>
      </div>
      
      {/* Connection Status */}
      <div className="mb-6 text-center">
        <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
          connectionStatus === 'connected' 
            ? 'bg-green-100 text-green-800' 
            : connectionStatus === 'connecting'
            ? 'bg-yellow-100 text-yellow-800'
            : connectionStatus === 'error'
            ? 'bg-red-100 text-red-800'
            : 'bg-gray-100 text-gray-800'
        }`}>
          <Activity className="w-4 h-4 mr-1" />
          {connectionStatus === 'connected' && 'Connected'}
          {connectionStatus === 'connecting' && 'Connecting...'}
          {connectionStatus === 'error' && 'Error'}
          {connectionStatus === 'disconnected' && 'Disconnected'}
        </div>
      </div>
      
      {/* Main Controls */}
      <div className="flex flex-col items-center space-y-6">
        
        {/* Listening Button */}
        <div className="relative">
          <button
            onClick={isListening ? stopListening : startListening}
            disabled={isProcessing && !isListening}
            className={`
              w-32 h-32 rounded-full flex items-center justify-center text-white font-semibold text-lg
              transition-all duration-200 transform hover:scale-105 active:scale-95
              ${isListening 
                ? 'bg-red-500 hover:bg-red-600 shadow-lg shadow-red-500/30' 
                : 'bg-green-500 hover:bg-green-600 shadow-lg shadow-green-500/30'
              }
              ${isProcessing && !isListening ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
            `}
          >
            {isListening ? (
              <MicOff className="w-8 h-8" />
            ) : (
              <Mic className="w-8 h-8" />
            )}
          </button>
          
          {/* Voice Activity Indicator */}
          {isListening && (
            <div className={`absolute -top-2 -right-2 w-6 h-6 rounded-full flex items-center justify-center ${
              voiceActivity ? 'bg-green-500 animate-pulse' : 'bg-gray-300'
            }`}>
              <Activity className="w-3 h-3 text-white" />
            </div>
          )}
          
          {/* Recording Timer */}
          {isListening && (
            <div className="absolute -bottom-10 left-1/2 transform -translate-x-1/2">
              <div className="bg-blue-500 text-white px-3 py-1 rounded-full text-sm font-medium">
                {formatTime(recordingTime)}
              </div>
            </div>
          )}
        </div>
        
        {/* Status Display */}
        <div className="text-center min-h-16 max-w-md">
          {error && (
            <div className="text-red-600 mb-3">
              <p className="font-medium">Error: {error}</p>
            </div>
          )}
          
          {isProcessing && (
            <div className="flex items-center justify-center space-x-2 text-blue-600 mb-3">
              <Zap className="w-4 h-4 animate-pulse" />
              <span className="font-medium">Processing audio...</span>
            </div>
          )}
          
          {isResponding && (
            <div className="flex items-center justify-center space-x-2 text-purple-600 mb-3">
              <Brain className="w-4 h-4 animate-pulse" />
              <span className="font-medium">AI is responding...</span>
            </div>
          )}
          
          {isListening && !isProcessing && !isResponding && (
            <div className="flex items-center justify-center space-x-2 text-green-600 mb-3">
              <Activity className="w-4 h-4" />
              <span className="font-medium">
                {voiceActivity ? 'Voice detected...' : 'Listening...'}
              </span>
            </div>
          )}
          
          {!isListening && !isProcessing && !isResponding && !error && (
            <div className="text-gray-500">
              <p>Click to start continuous listening</p>
            </div>
          )}
        </div>
        
        {/* Transcription Display */}
        {(partialTranscription || currentTranscription) && (
          <div className="w-full max-w-2xl space-y-3">
            {partialTranscription && (
              <div className="p-3 bg-yellow-50 rounded-lg border border-yellow-200">
                <p className="text-sm text-yellow-800">
                  <strong>Partial:</strong> "{partialTranscription}"
                </p>
              </div>
            )}
            
            {currentTranscription && (
              <div className="p-3 bg-blue-50 rounded-lg border border-blue-200">
                <p className="text-sm text-blue-800">
                  <strong>You said:</strong> "{currentTranscription}"
                </p>
              </div>
            )}
          </div>
        )}
        
        {/* AI Response Display */}
        {aiResponse && (
          <div className="w-full max-w-2xl">
            <div className="p-3 bg-green-50 rounded-lg border border-green-200">
              <p className="text-sm text-green-800">
                <strong>AI:</strong> {aiResponse}
              </p>
            </div>
          </div>
        )}
        
        {/* Provider Info */}
        <div className="flex items-center space-x-2 bg-gray-50 px-4 py-2 rounded-lg">
          <Brain className="w-4 h-4 text-gray-600" />
          <span className="text-sm text-gray-600">Current AI:</span>
          <span className="text-sm font-medium text-gray-900 capitalize">
            {currentProvider} / {currentModel}
          </span>
        </div>
      </div>
    </div>
  );
};

export default ContinuousListening;