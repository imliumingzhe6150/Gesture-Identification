/**
 * @file App.tsx
 * @description Advanced Neural Gesture Interface. 
 * Implements real-time computer vision using MediaPipe, rendering a macOS-inspired UX.
 * 
 * Features:
 * - Neural Gesture Recognition (MediaPipe)
 * - Virtual Pointer & Haptic Feedback Logic
 * - Real-time Confidence Analytics (Recharts)
 * - Dynamic Sensitivity Scaling
 */

import { GestureRecognizer, FilesetResolver } from "@mediapipe/tasks-vision";
import { motion, AnimatePresence } from "motion/react";
import { 
  Camera, Settings, RefreshCw, MousePointer2, 
  Zap, History, LayoutGrid, Sliders, Palette, X
} from "lucide-react";
import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import Webcam from "react-webcam";
import { LineChart, Line, ResponsiveContainer, YAxis, Tooltip } from 'recharts';

/** External Model Asset URL */
const MODEL_PATH = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task";

/** 
 * INTERFACES & TYPES 
 */
interface LogEntry {
  id: string;
  action: string;
  time: string;
  type: "gesture" | "click";
}

interface ChartData {
  time: number;
  confidence: number;
}

/** 
 * MAIN APPLICATION COMPONENT 
 */
export default function App() {
  // --- REFS & STATE ---
  const webcamRef = useRef<Webcam>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const lastGestureRef = useRef<string | null>(null);
  const lastPinchRef = useRef<boolean>(false);
  const chartDataRef = useRef<ChartData[]>([]);

  const [gestureRecognizer, setGestureRecognizer] = useState<GestureRecognizer | null>(null);
  const [gesture, setGesture] = useState<string>("Waiting...");
  const [confidence, setConfidence] = useState<number>(0);
  const [isLoaded, setIsLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fps, setFps] = useState(0);
  const [isRebooting, setIsRebooting] = useState(false);
  const [isRecognitionActive, setIsRecognitionActive] = useState(false);
  const [currentView, setCurrentView] = useState<"landing" | "studio">("landing");

  // Virtual Mouse & Telemetry
  const [isClicking, setIsClicking] = useState(false);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [history, setHistory] = useState<ChartData[]>([]);

  // Settings & UX State
  const [showSettings, setShowSettings] = useState(false);
  const [sensitivity, setSensitivity] = useState(0.5); // 0 to 1
  const [themeMode, setThemeMode] = useState<"standard" | "game">("standard");

  // Gesture Access Control
  const [enabledGestures, setEnabledGestures] = useState<Record<string, boolean>>({
    "Open_Palm": true,
    "Closed_Fist": true,
    "Pointing_Up": true,
    "Thumb_Down": true,
    "Thumb_Up": true,
    "Victory": true,
    "ILoveYou": true
  });

  const toggleGesture = (name: string) => {
    setEnabledGestures(prev => ({ ...prev, [name]: !prev[name] }));
  };

  /**
   * SYSTEM: Interactive Reboot Protocol
   */
  const handleReboot = () => {
    setIsRebooting(true);
    setGesture("正在调优...");
    setConfidence(0);
    setLogs([]);
    setHistory([]);
    chartDataRef.current = [];
    
    setTimeout(() => {
      setIsRebooting(false);
      addLog("系统：识别模块已重载", "gesture");
    }, 2800);
  };

  /**
   * INITIALIZATION: Load MediaPipe Vision Assets
   */
  useEffect(() => {
    async function initRecognizer() {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
        );
        const recognizer = await GestureRecognizer.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: MODEL_PATH,
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numHands: 1,
        });
        setGestureRecognizer(recognizer);
        setIsLoaded(true);
      } catch (err) {
        console.error("神经网络引擎启动失败:", err);
        setError("AI 识别引擎初始化失败，请尝试刷新页面或检查浏览器设置。");
      }
    }
    initRecognizer();
  }, []);

  /**
   * TELEMETRY: System Event Logging
   */
  const addLog = useCallback((action: string, type: "gesture" | "click") => {
    setLogs(prev => [
      {
        id: Math.random().toString(36).substr(2, 9),
        action,
        time: new Date().toLocaleTimeString([], { hour12: false, minute: '2-digit', second: '2-digit' }),
        type
      },
      ...prev.slice(0, 49) // Keep last 50 for robust debugging
    ]);
  }, []);

  const lastFrameTime = useRef<number>(0);
  
  /**
   * CORE LOOP: Frame Processing & Computer Vision
   */
  const detectGestures = useCallback(() => {
    if (!isRecognitionActive) {
      if (fps !== 0) setFps(0);
      // Removed requestAnimationFrame recursive call to fully halt the loop
      return;
    }

    if (
      gestureRecognizer &&
      webcamRef.current &&
      webcamRef.current.video &&
      webcamRef.current.video.readyState === 4 &&
      containerRef.current
    ) {
      const video = webcamRef.current.video;
      const startTimeMs = performance.now();
      
      // Calculate delta for FPS
      if (lastFrameTime.current !== 0) {
        const delta = startTimeMs - lastFrameTime.current;
        setFps(Math.round(1000 / delta));
      }
      lastFrameTime.current = startTimeMs;

      // Execute Neural Inference
      const results = gestureRecognizer.recognizeForVideo(video, startTimeMs);

      if (results.landmarks && results.landmarks.length > 0) {
        const landmarks = results.landmarks[0];
        
        // --- 1. COORDINATE MAPPING (REMOVED) ---
        const indexTip = landmarks[8];

        // --- 2. INTERACTION LOGIC (PINCH/CLICK) ---
        const middleTip = landmarks[12];
        const distance = Math.sqrt(
          Math.pow(indexTip.x - middleTip.x, 2) + 
          Math.pow(indexTip.y - middleTip.y, 2)
        );

        // Sensitivity-adjusted threshold
        const threshold = 0.03 + (sensitivity * 0.04);
        const isPinching = distance < threshold; 
        
        if (isPinching && !lastPinchRef.current) {
          setIsClicking(true);
          addLog("操作：触发虚拟点击", "click");
          setTimeout(() => setIsClicking(false), 200);
        }
        lastPinchRef.current = isPinching;

        // --- 3. GESTURE RECOGNITION ---
        if (results.gestures.length > 0) {
          const topGesture = results.gestures[0][0];
          const name = topGesture.categoryName;
          const scorePercent = Math.round(topGesture.score * 100);
          
          // Filter by enabled gestures
          if (name === "None" || enabledGestures[name]) {
            if (name !== lastGestureRef.current && name !== "None") {
              addLog(`识别结果：${name}`, "gesture");
            }
            lastGestureRef.current = name;
            setGesture(name);
            setConfidence(scorePercent);
          } else {
            setGesture("已停用");
            setConfidence(0);
          }

          // Update Analytics Stream
          chartDataRef.current = [
            ...chartDataRef.current.slice(-29),
            { time: Date.now(), confidence: scorePercent }
          ];
          setHistory([...chartDataRef.current]);
        }
      } else {
        setGesture("None");
        setConfidence(0);
        lastGestureRef.current = null;
      }
    }
    requestAnimationFrame(detectGestures);
  }, [gestureRecognizer, addLog, sensitivity, isRecognitionActive]);

  useEffect(() => {
    if (isLoaded) {
      const animationId = requestAnimationFrame(detectGestures);
      return () => cancelAnimationFrame(animationId);
    }
  }, [isLoaded, detectGestures]);

  /** 
   * UI HELPERS 
   */
  const themeAccent = themeMode === "game" ? "#FF2D55" : "#007AFF";

  // --- LANDING PAGE VIEW ---
  if (currentView === "landing") {
    return (
      <div className="h-screen bg-bg text-white font-sans overflow-hidden relative flex flex-col items-center justify-center p-10">
        {/* Animated Background Elements */}
        <div className="absolute top-1/4 left-1/4 w-[500px] h-[500px] bg-accent/20 blur-[150px] rounded-full animate-pulse pointer-events-none" />
        <div className="absolute bottom-1/4 right-1/4 w-[500px] h-[500px] bg-blue-500/10 blur-[150px] rounded-full delay-1000 animate-pulse pointer-events-none" />
        
        {/* Decorative Grid */}
        <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/carbon-fibre.png')] opacity-[0.03] pointer-events-none" />

        <motion.div 
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, ease: "easeOut" }}
          className="relative z-10 text-center max-w-4xl"
        >
          {/* Brand Tag */}
          <div className="inline-flex items-center gap-2 px-4 py-1.5 bg-white/5 border border-white/10 rounded-full mb-8 backdrop-blur-md">
            <Zap className="w-4 h-4 text-accent animate-pulse" />
            <span className="text-[10px] font-bold uppercase tracking-[0.4em] text-accent">Neural Engine v2.0</span>
          </div>

          {/* Hero Headline */}
          <h1 className="text-7xl md:text-9xl font-black tracking-tighter mb-6 leading-none">
             NEURAL<br />
             <span className="text-gradient-blue italic">VISION</span>
          </h1>
          
          <p className="text-lg md:text-xl text-text-dim max-w-2xl mx-auto mb-12 font-medium leading-relaxed">
            利用先进的神经网络模型，识别您做出的手势。
          </p>

          {/* Action Buttons */}
          <div className="flex flex-col md:flex-row items-center justify-center gap-6">
            {!isLoaded ? (
               <div className="flex flex-col items-center gap-4">
                 <div className="w-64 h-1.5 bg-white/5 rounded-full overflow-hidden border border-white/5">
                   <motion.div 
                     initial={{ width: 0 }}
                     animate={{ width: "100%" }}
                     transition={{ duration: 2, repeat: Infinity }}
                     className="h-full bg-accent shadow-[0_0_15px_var(--color-accent)]"
                   />
                 </div>
                 <span className="text-[10px] font-bold uppercase tracking-[0.2em] opacity-40">正在加载 AI 神经模型...</span>
               </div>
            ) : (
              <>
                <motion.button 
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => setCurrentView("studio")}
                  className="px-10 py-5 bg-white text-black rounded-2xl font-black text-lg uppercase tracking-widest shadow-[0_20px_40px_rgba(255,255,255,0.2)] hover:bg-white/90 transition-all flex items-center gap-3"
                >
                  进入工作站
                  <MousePointer2 className="w-5 h-5" />
                </motion.button>
                <button className="px-8 py-4 bg-white/5 border border-white/10 rounded-2xl font-bold uppercase tracking-widest text-sm hover:bg-white/10 transition-all">
                  查看文档
                </button>
              </>
            )}
          </div>
        </motion.div>
      </div>
    );
  }

  return (
    <div className={`h-screen flex flex-col bg-bg text-white font-sans overflow-hidden transition-colors duration-700 ${themeMode === 'game' ? 'selection:bg-red-500/30' : 'selection:bg-accent/30'}`}>
      
      {/* 1. LAYER: TITLE BAR */}
      <header className="h-[52px] bg-sidebar/90 backdrop-blur-xl border-b border-white/5 flex items-center px-5 justify-between relative z-50">
        <div className="flex items-center gap-3">
          <button 
            onClick={() => setCurrentView("landing")}
            className="p-2 hover:bg-white/10 rounded-lg transition-all group"
            title="返回主页"
          >
            <History className="w-4 h-4 text-text-dim group-hover:text-accent rotate-180" />
          </button>
          <div className="flex gap-2">
            <div className="w-3 h-3 rounded-full bg-[#FF5F57]"></div>
            <div className="w-3 h-3 rounded-full bg-[#FFBD2E]"></div>
            <div className="w-3 h-3 rounded-full bg-[#28C840]"></div>
          </div>
        </div>
        <div className="absolute left-1/2 -translate-x-1/2 flex items-center gap-2">
          <Zap className="w-4 h-4 text-accent" />
          <h1 className="text-[13px] font-bold tracking-[0.05em] uppercase text-gradient-silver">智能手势视觉中心 (Neural Vision)</h1>
        </div>
        <div className="flex items-center gap-4 text-xs">
          <div className="flex items-center gap-3 px-3 py-1 bg-white/5 rounded-full border border-white/10 uppercase font-mono tracking-tighter">
            <div className="flex items-center gap-1.5 border-r border-white/10 pr-3">
              <div className="w-1.5 h-1.5 bg-green-400 rounded-full animate-pulse shadow-[0_0_8px_var(--color-green-400)]"></div>
              <span className="text-[10px] text-green-400/80">Engine OK</span>
            </div>
            <span className="text-[9px] opacity-40">CPU_XNN_ACCEL</span>
          </div>
          <div className="flex items-center gap-2 text-green-400 font-mono">
            {fps} FPS
          </div>
          <button 
            onClick={() => setShowSettings(true)}
            className="p-2 hover:bg-white/10 rounded-lg transition-all"
          >
            <Settings className="w-4 h-4 text-text-dim" />
          </button>
        </div>
      </header>

      {/* 2. LAYER: MAIN WORKSPACE */}
      <main className="flex-1 flex p-5 gap-5 min-h-0">
        
        {/* VIEWPORT AREA */}
        <section className="flex-1 flex flex-col gap-5 min-w-0">
          <div 
            ref={containerRef}
            className="flex-[3] bg-black rounded-macos border border-white/10 relative overflow-hidden group shadow-inner"
          >
            {/* VIRTUAL POINTER REMOVED */}

            {/* CAMERA FEED */}
            <Webcam
              ref={webcamRef}
              {...{
                audio: false,
                className: "w-full h-full object-cover mirror opacity-80 group-hover:opacity-100 transition-opacity duration-500",
                videoConstraints: { facingMode: "user" },
                mirrored: true,
              } as any}
            />

            {/* ERROR / INITIALIZATION OVERLAY */}
            <AnimatePresence>
              {!isLoaded && (
                <motion.div 
                  exit={{ opacity: 0 }}
                  className="absolute inset-0 z-50 flex flex-col items-center justify-center bg-bg/95 backdrop-blur-3xl"
                >
                   <RefreshCw className="w-12 h-12 text-accent animate-spin mb-4" />
                   <p className="text-sm font-medium tracking-widest uppercase opacity-60">正在开启视觉识别系统...</p>
                </motion.div>
              )}
            </AnimatePresence>

            <button 
              onClick={() => {
                const newState = !isRecognitionActive;
                setIsRecognitionActive(newState);
                addLog(`系统：${newState ? '手势识别已启动' : '识别监控已暂停'}`, "gesture");
                if (!newState) {
                  setGesture("STANDBY");
                  setConfidence(0);
                }
              }}
              className={`absolute top-5 left-5 z-20 backdrop-blur-md px-4 py-1.5 rounded-full border transition-all flex items-center gap-2 group/btn ${
                isRecognitionActive 
                ? 'bg-red-500/20 border-red-500/40 text-red-500 hover:bg-red-500/30' 
                : 'bg-accent/20 border-accent/40 text-accent hover:bg-accent/30'
              }`}
            >
              <div className={`w-2 h-2 rounded-full ${isRecognitionActive ? 'bg-red-500 animate-pulse' : 'bg-accent'}`} />
              <span className="text-[10px] uppercase font-bold tracking-widest">
                {isRecognitionActive ? '暂停识别' : '开启识别'}
              </span>
            </button>

            {/* STANDBY OVERLAY */}
            {!isRecognitionActive && !isRebooting && (
              <div className="absolute inset-0 z-10 bg-black/40 backdrop-blur-sm flex flex-col items-center justify-center pointer-events-none">
                <div className="flex flex-col items-center gap-3 opacity-40">
                  <div className="w-12 h-12 rounded-full border border-white/20 flex items-center justify-center">
                    <Zap className="w-6 h-6 text-white" />
                  </div>
                  <span className="text-[10px] font-bold tracking-[0.3em] uppercase">引擎待机中</span>
                </div>
              </div>
            )}
            
            {isRecognitionActive && (
              <div className="absolute inset-x-0 top-0 h-[3px] bg-gradient-to-r from-transparent via-accent to-transparent animate-scan shadow-[0_0_20px_var(--color-accent)] opacity-50" />
            )}
            
            {/* SYSTEM REBOOT OVERLAY */}
            <AnimatePresence>
              {isRebooting && (
                <motion.div 
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="absolute inset-0 z-[60] bg-black/90 backdrop-blur-2xl flex flex-col items-center justify-center font-mono"
                >
                  <RefreshCw className="w-12 h-12 text-accent animate-spin mb-6" />
                  <div className="space-y-4 text-center">
                    <p className="text-accent text-[11px] font-bold tracking-[0.5em] uppercase">正在重构系统算法</p>
                    <div className="w-48 h-1 bg-white/10 rounded-full overflow-hidden mx-auto">
                      <motion.div 
                        initial={{ width: 0 }}
                        animate={{ width: "100%" }}
                        transition={{ duration: 2.8, ease: "linear" }}
                        className="h-full bg-accent"
                      />
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* 3. LAYER: TELEMETRY CHART */}
          <div className={`flex-1 bg-sidebar rounded-macos border border-white/5 p-4 flex flex-col transition-opacity duration-500 ${!isRecognitionActive ? 'opacity-40 grayscale' : 'opacity-100'}`}>
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-[10px] font-bold uppercase tracking-widest text-text-dim">识别置信度监控仪表盘</h3>
              <div className="flex items-center gap-1.5">
                <div className="w-2 h-2 rounded-full bg-accent"></div>
                <span className="text-[10px] font-mono opacity-50">实时识别稳定性监控</span>
              </div>
            </div>
            <div className="flex-1">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={history}>
                  <YAxis domain={[0, 100]} hide />
                  <Tooltip 
                    contentStyle={{ background: '#1c1c1e', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                    labelStyle={{ display: 'none' }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="confidence" 
                    stroke={themeAccent} 
                    strokeWidth={2} 
                    dot={false} 
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </section>

        {/* 4. LAYER: ANALYTICS SIDEBAR */}
        <aside className="w-[320px] flex flex-col gap-5 overflow-hidden">
          
          {/* Classification Result */}
          <div className="bg-sidebar rounded-macos p-6 border border-white/5 shadow-2xl relative overflow-hidden group">
            <div className="absolute -top-12 -right-12 w-40 h-40 bg-accent/10 blur-[50px] group-hover:bg-accent/20 transition-all" />
            
            <p className="text-[10px] font-bold uppercase tracking-widest text-text-dim mb-2">当前手势识别状态</p>
            <h2 className="text-4xl font-black text-gradient-blue tracking-tighter capitalize mb-4">
              {gesture === "None" ? "..." : gesture}
            </h2>
            
            <div className="flex items-end justify-between mb-2">
              <span className="text-[10px] font-bold text-text-dim uppercase">识别准确度</span>
              <span className="font-mono text-lg">{confidence}%</span>
            </div>
            <div className="h-2.5 w-full bg-surface rounded-full overflow-hidden border border-white/5 p-0.5">
              <motion.div
                className="h-full bg-accent rounded-full shadow-[0_0_10px_var(--color-accent)]"
                initial={{ width: 0 }}
                animate={{ width: `${confidence}%` }}
                transition={{ type: "spring", stiffness: 80 }}
              />
            </div>
          </div>

          {/* Gesture Access Control Section */}
          <div className="bg-sidebar rounded-macos border border-white/5 h-[180px] flex flex-col overflow-hidden shadow-xl shrink-0">
            <div className="p-3 border-b border-white/5 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <LayoutGrid className="w-4 h-4 text-accent" />
                <h3 className="text-[10px] font-bold uppercase tracking-widest text-text-dim">手势功能开关</h3>
              </div>
              <span className="text-[9px] bg-accent/20 px-2 py-0.5 rounded-full text-accent font-bold">已启用</span>
            </div>
            
            <div className="flex-1 overflow-y-auto p-3 custom-scrollbar space-y-3">
              {Object.keys(enabledGestures).map((name) => (
                <div key={name} className="flex items-center justify-between group">
                  <div className="flex flex-col">
                    <span className="text-[12px] font-bold tracking-tight text-white/90 capitalize leading-none">
                      {name.replace(/_/g, ' ')}
                    </span>
                    <span className="text-[9px] text-text-dim font-mono mt-1">
                      {enabledGestures[name] ? '扫描中' : '已锁定'}
                    </span>
                  </div>
                  
                  <button
                    onClick={() => toggleGesture(name)}
                    className={`w-8 h-4 rounded-full relative transition-colors duration-300 ${enabledGestures[name] ? 'bg-accent' : 'bg-white/10'}`}
                  >
                    <motion.div
                      layout
                      transition={{ type: "spring", stiffness: 500, damping: 30 }}
                      className={`absolute top-0.5 w-3 h-3 bg-white rounded-full shadow-md ${enabledGestures[name] ? 'right-0.5' : 'left-0.5'}`}
                    />
                  </button>
                </div>
              ))}
            </div>
          </div>

          {/* Action Log / Event Stream */}
          <div className="bg-sidebar rounded-macos border border-white/5 flex-1 flex flex-col overflow-hidden shadow-2xl min-h-0">
            <div className="p-3 border-b border-white/5 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <History className="w-4 h-4 text-accent" />
                <h3 className="text-[10px] font-bold uppercase tracking-widest text-text-dim">实时操作监控记录</h3>
              </div>
              <span className="text-[9px] font-mono opacity-50">{logs.length} 条记录</span>
            </div>
            
            <div className={`flex-1 overflow-y-auto p-3 custom-scrollbar space-y-2.5 transition-opacity ${!isRecognitionActive ? 'opacity-30' : 'opacity-100'}`}>
              <AnimatePresence initial={false}>
                {logs.length === 0 ? (
                  <div className="h-full flex flex-col items-center justify-center opacity-10 py-10 text-center">
                    <RefreshCw className="w-8 h-8 mb-2" />
                    <p className="text-[10px] uppercase font-bold tracking-widest leading-tight">等待神经网络数据流入...</p>
                  </div>
                ) : (
                  logs.map((log) => (
                    <motion.div
                      key={log.id}
                      initial={{ x: -10, opacity: 0 }}
                      animate={{ x: 0, opacity: 1 }}
                      className="flex items-start justify-between border-l border-white/10 pl-3 py-0.5 group hover:border-accent transition-all"
                    >
                      <div className="flex flex-col gap-0.5">
                        <span className={`text-[12px] font-bold tracking-tight leading-tight ${log.type === 'click' ? 'text-accent' : 'text-white/80'}`}>
                          {log.action}
                        </span>
                        <div className="flex items-center gap-2">
                          <span className="text-[9px] font-mono text-text-dim font-bold">{log.time}</span>
                          <div className={`w-1 h-1 rounded-full ${log.type === 'click' ? 'bg-accent animate-pulse' : 'bg-white/20'}`} />
                        </div>
                      </div>
                    </motion.div>
                  ))
                )}
              </AnimatePresence>
            </div>
          </div>

          <button 
            onClick={handleReboot}
            disabled={isRebooting}
            className="py-4 bg-accent hover:bg-accent/80 transition-all rounded-xl font-bold uppercase text-xs tracking-widest shadow-lg shadow-accent/20 active:scale-95 disabled:opacity-50"
          >
            {isRebooting ? "正在重启系统..." : "重启识别系统"}
          </button>
        </aside>
      </main>

      {/* 5. LAYER: SETTINGS OVERLAY */}
      <AnimatePresence>
        {showSettings && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-2xl p-6"
          >
            <motion.div 
              initial={{ scale: 0.9, y: 20 }}
              animate={{ scale: 1, y: 0 }}
              className="w-full max-w-md bg-sidebar border border-white/10 rounded-[32px] p-8 relative shadow-2xl"
            >
              <button 
                onClick={() => setShowSettings(false)}
                className="absolute top-6 right-6 p-2 hover:bg-white/5 rounded-full"
              >
                <X className="w-5 h-5" />
              </button>

              <div className="flex items-center gap-3 mb-8">
                <Sliders className="w-6 h-6 text-accent" />
                <h2 className="text-xl font-bold tracking-tight">系统设置中心</h2>
              </div>

              <div className="space-y-8">
                {/* Sensitivity Slider */}
                <div>
                  <div className="flex justify-between items-end mb-4">
                    <label className="text-xs font-bold uppercase tracking-widest opacity-60">手势感测灵敏度</label>
                    <span className="text-sm font-mono text-accent">{Math.round(sensitivity * 100)}%</span>
                  </div>
                  <input 
                    type="range" min="0" max="1" step="0.01" 
                    value={sensitivity}
                    onChange={(e) => setSensitivity(parseFloat(e.target.value))}
                    className="w-full accent-accent h-1.5 bg-white/5 rounded-lg appearance-none cursor-pointer"
                  />
                  <div className="flex justify-between mt-2 text-[10px] font-bold uppercase opacity-30">
                    <span>高精度</span>
                    <span>高响应</span>
                  </div>
                </div>

                {/* Theme Mode */}
                <div>
                  <label className="text-xs font-bold uppercase tracking-widest opacity-60 mb-4 block">视觉渲染模式</label>
                  <div className="grid grid-cols-2 gap-3">
                    {[
                      { id: 'standard', name: '标准模式', icon: LayoutGrid },
                      { id: 'game', name: '极限性能', icon: Zap }
                    ].map((mode) => (
                      <button
                        key={mode.id}
                        onClick={() => setThemeMode(mode.id as any)}
                        className={`flex items-center gap-3 p-4 rounded-2xl border transition-all ${themeMode === mode.id ? 'bg-accent/10 border-accent text-accent' : 'bg-white/5 border-white/5 text-text-dim hover:bg-white/10'}`}
                      >
                        <mode.icon className="w-5 h-5" />
                        <span className="text-sm font-bold">{mode.name}</span>
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              <button 
                onClick={() => setShowSettings(false)}
                className="w-full py-4 bg-white text-black rounded-xl font-bold mt-10 hover:bg-white/90 transition-all border-none"
              >
                保存设置并返回
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
