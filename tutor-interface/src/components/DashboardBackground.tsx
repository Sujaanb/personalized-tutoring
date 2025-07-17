
import React from 'react';

const DashboardBackground = () => {
  return (
    <div className="fixed inset-0 overflow-hidden pointer-events-none">
      {/* Base gradient background */}
      <div className="absolute inset-0 bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900" />
      
      {/* Dynamic gradient overlay */}
      <div className="absolute inset-0 bg-gradient-to-tr from-emerald-900/20 via-transparent to-cyan-900/20" />
      
      {/* Animated learning symbols */}
      <div className="absolute inset-0">
        {/* Brain networks */}
        <div className="absolute top-20 left-10 w-16 h-16 border-2 border-emerald-400/20 rounded-full animate-neural-pulse">
          <div className="absolute inset-2 border border-emerald-400/30 rounded-full animate-pulse-scale" />
          <div className="absolute inset-4 bg-emerald-400/20 rounded-full animate-core-pulse" />
        </div>
        
        <div className="absolute top-32 right-20 w-12 h-12 border-2 border-teal-400/20 rounded-full animate-neural-pulse" style={{ animationDelay: '1s' }}>
          <div className="absolute inset-2 border border-teal-400/30 rounded-full animate-pulse-scale" />
        </div>
        
        <div className="absolute bottom-40 left-16 w-14 h-14 border-2 border-cyan-400/20 rounded-full animate-neural-pulse" style={{ animationDelay: '2s' }}>
          <div className="absolute inset-2 border border-cyan-400/30 rounded-full animate-pulse-scale" />
        </div>

        {/* Knowledge connection lines */}
        <div className="absolute top-1/4 left-1/4 w-px h-32 bg-gradient-to-b from-transparent via-emerald-400/20 to-transparent animate-data-flow-vertical" />
        <div className="absolute top-1/3 right-1/3 w-24 h-px bg-gradient-to-r from-transparent via-teal-400/20 to-transparent animate-data-flow" style={{ animationDelay: '0.5s' }} />
        <div className="absolute bottom-1/3 left-1/2 w-px h-24 bg-gradient-to-b from-transparent via-cyan-400/20 to-transparent animate-data-flow-vertical" style={{ animationDelay: '1s' }} />
        
        {/* Learning nodes */}
        <div className="absolute top-1/2 left-1/4 w-3 h-3 bg-emerald-400/40 rounded-full animate-particle-glow" />
        <div className="absolute top-1/3 right-1/4 w-2 h-2 bg-teal-400/40 rounded-full animate-particle-glow" style={{ animationDelay: '0.7s' }} />
        <div className="absolute bottom-1/4 left-1/3 w-4 h-4 bg-cyan-400/40 rounded-full animate-particle-glow" style={{ animationDelay: '1.4s' }} />
        
        {/* Floating learning symbols */}
        <div className="absolute top-1/4 right-1/4 text-2xl text-emerald-400/30 animate-symbol-float">📚</div>
        <div className="absolute bottom-1/3 left-1/3 text-xl text-teal-400/30 animate-symbol-float" style={{ animationDelay: '0.5s' }}>🎓</div>
        <div className="absolute top-1/2 right-1/3 text-2xl text-cyan-400/30 animate-symbol-float" style={{ animationDelay: '1s' }}>🧠</div>
        <div className="absolute bottom-1/4 right-1/4 text-xl text-emerald-400/30 animate-symbol-float" style={{ animationDelay: '1.5s' }}>💡</div>
        
        {/* Flowing data streams */}
        <div className="absolute top-1/2 left-0 w-full h-px">
          <div className="w-20 h-px bg-gradient-to-r from-transparent via-emerald-400/20 to-transparent animate-stream-flow" />
        </div>
        <div className="absolute top-2/3 left-0 w-full h-px">
          <div className="w-16 h-px bg-gradient-to-r from-transparent via-teal-400/20 to-transparent animate-stream-flow" style={{ animationDelay: '1s' }} />
        </div>
        
        {/* Interactive learning grid */}
        <div className="absolute inset-0 opacity-10">
          <div className="w-full h-full" style={{
            backgroundImage: 'linear-gradient(rgba(6, 182, 212, 0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(6, 182, 212, 0.1) 1px, transparent 1px)',
            backgroundSize: '60px 60px'
          }} />
        </div>
        
        {/* Scanning effect */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-emerald-400/30 to-transparent animate-grid-scan" />
          <div className="absolute top-0 left-0 w-1 h-full bg-gradient-to-b from-transparent via-teal-400/30 to-transparent animate-grid-scan-vertical" style={{ animationDelay: '2s' }} />
        </div>
        
        {/* Orbiting knowledge particles */}
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
          <div className="relative w-2 h-2">
            <div className="absolute w-2 h-2 bg-emerald-400/50 rounded-full animate-orbit-1" />
            <div className="absolute w-1.5 h-1.5 bg-teal-400/50 rounded-full animate-orbit-2" />
            <div className="absolute w-1 h-1 bg-cyan-400/50 rounded-full animate-orbit-3" />
          </div>
        </div>
        
        {/* Ambient light effects */}
        <div className="absolute top-20 right-20 w-32 h-32 bg-emerald-400/5 rounded-full blur-xl animate-gradient-shift" />
        <div className="absolute bottom-20 left-20 w-40 h-40 bg-teal-400/5 rounded-full blur-xl animate-gradient-shift" style={{ animationDelay: '2s' }} />
        <div className="absolute top-1/2 right-1/3 w-24 h-24 bg-cyan-400/5 rounded-full blur-xl animate-gradient-shift" style={{ animationDelay: '4s' }} />
      </div>
    </div>
  );
};

export default DashboardBackground;
