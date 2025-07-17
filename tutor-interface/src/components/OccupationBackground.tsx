
import React, { useState, useEffect, useRef } from 'react';

const OccupationBackground = () => {
  const [particles, setParticles] = useState([]);
  const [careerPaths, setCareerPaths] = useState([]);
  const containerRef = useRef(null);

  // Generate floating particles with career-focused colors
  useEffect(() => {
    const generateParticles = () => {
      const newParticles = [];
      for (let i = 0; i < 40; i++) {
        newParticles.push({
          id: i,
          x: Math.random() * 100,
          y: Math.random() * 100,
          size: Math.random() * 6 + 3,
          opacity: Math.random() * 0.7 + 0.3,
          speed: Math.random() * 0.1 + 0.03,
          direction: Math.random() * Math.PI * 2,
          color: ['emerald', 'blue', 'violet', 'amber', 'rose'][Math.floor(Math.random() * 5)],
          pulse: Math.random() * 3 + 2
        });
      }
      setParticles(newParticles);
    };

    const generateCareerPaths = () => {
      const paths = [];
      for (let i = 0; i < 12; i++) {
        paths.push({
          id: i,
          x: Math.random() * 100,
          y: Math.random() * 100,
          width: Math.random() * 150 + 80,
          height: Math.random() * 3 + 1.5,
          angle: Math.random() * 180,
          speed: Math.random() * 1.5 + 0.8,
          color: ['emerald', 'blue', 'violet', 'amber'][Math.floor(Math.random() * 4)]
        });
      }
      setCareerPaths(paths);
    };

    generateParticles();
    generateCareerPaths();

    // Animate particles with smoother movement
    const interval = setInterval(() => {
      setParticles(prev => prev.map(p => ({
        ...p,
        x: (p.x + Math.cos(p.direction) * p.speed) % 100,
        y: (p.y + Math.sin(p.direction) * p.speed) % 100,
        direction: p.direction + 0.005
      })));

      setCareerPaths(prev => prev.map(s => ({
        ...s,
        x: (s.x + s.speed * 0.08) % 100,
        angle: s.angle + 0.3
      })));
    }, 60);

    return () => clearInterval(interval);
  }, []);

  return (
    <div ref={containerRef} className="absolute inset-0 w-full h-full bg-gradient-to-br from-slate-900 via-slate-800 to-indigo-900 overflow-hidden">
      {/* Professional gradient overlay */}
      <div className="absolute inset-0 bg-gradient-to-br from-emerald-900/15 via-blue-900/20 to-violet-900/15 animate-gradient-shift"></div>
      
      {/* Career milestone nodes */}
      <div className="absolute top-20 left-20 w-16 h-16 border-3 border-emerald-400/40 rounded-xl animate-pulse-scale bg-emerald-500/10"></div>
      <div className="absolute top-40 right-32 w-12 h-12 bg-gradient-to-br from-blue-500/25 to-violet-500/25 rounded-lg animate-float-rotate"></div>
      <div className="absolute bottom-32 left-24 w-20 h-20 border-3 border-amber-400/40 rounded-full animate-float-reverse bg-amber-500/10"></div>
      <div className="absolute bottom-40 right-20 w-14 h-14 bg-gradient-to-br from-rose-500/25 to-pink-500/25 transform rotate-12 animate-spin-slow"></div>
      
      {/* Professional network visualization */}
      <div className="absolute top-1/3 left-1/3 w-28 h-28 animate-neural-pulse">
        <div className="absolute inset-0 border-2 border-emerald-300/25 rounded-lg"></div>
        <div className="absolute inset-3 border-2 border-blue-300/35 rounded-lg animate-ping"></div>
        <div className="absolute inset-6 border-2 border-violet-300/45 rounded-lg animate-pulse"></div>
        <div className="absolute top-1/2 left-0 w-full h-0.5 bg-gradient-to-r from-transparent via-emerald-400/60 to-transparent animate-data-flow"></div>
        <div className="absolute left-1/2 top-0 w-0.5 h-full bg-gradient-to-b from-transparent via-blue-400/60 to-transparent animate-data-flow-vertical"></div>
      </div>

      {/* Career progression orbits */}
      <div className="absolute top-24 right-1/3 w-10 h-10 bg-gradient-to-br from-emerald-400/35 to-green-500/35 rounded-lg animate-orbit-1 border-2 border-emerald-300/60">
        <div className="absolute inset-2 bg-white/25 rounded-sm animate-pulse"></div>
      </div>
      <div className="absolute bottom-28 left-1/4 w-8 h-8 bg-gradient-to-br from-blue-400/35 to-cyan-500/35 rounded-lg animate-orbit-2 border-2 border-blue-300/60">
        <div className="absolute inset-2 bg-white/25 rounded-sm animate-pulse"></div>
      </div>
      <div className="absolute top-1/2 right-12 w-12 h-12 bg-gradient-to-br from-violet-400/35 to-purple-500/35 rounded-lg animate-orbit-3 border-2 border-violet-300/60">
        <div className="absolute inset-2 bg-white/25 rounded-sm animate-pulse"></div>
      </div>

      {/* Dynamic particles with professional colors */}
      {particles.map(particle => (
        <div
          key={particle.id}
          className={`absolute rounded-full animate-particle-glow`}
          style={{
            left: `${particle.x}%`,
            top: `${particle.y}%`,
            width: `${particle.size}px`,
            height: `${particle.size}px`,
            opacity: particle.opacity,
            background: `radial-gradient(circle, var(--${particle.color}-400), transparent)`,
            boxShadow: `0 0 ${particle.size * 1.5}px var(--${particle.color}-400)`,
            animationDuration: `${particle.pulse}s`
          }}
        />
      ))}

      {/* Career path connections */}
      {careerPaths.map(path => (
        <div
          key={path.id}
          className="absolute animate-stream-flow"
          style={{
            left: `${path.x}%`,
            top: `${path.y}%`,
            width: `${path.width}px`,
            height: `${path.height}px`,
            background: `linear-gradient(90deg, transparent, var(--${path.color}-400), transparent)`,
            transform: `rotate(${path.angle}deg)`,
            opacity: 0.5
          }}
        >
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-pulse"></div>
        </div>
      ))}

      {/* Professional grid overlay */}
      <div className="absolute inset-0 opacity-8">
        <div className="w-full h-full bg-gradient-to-r from-transparent via-emerald-500/15 to-transparent animate-grid-scan"></div>
        <div className="absolute inset-0 w-full h-full bg-gradient-to-b from-transparent via-blue-500/15 to-transparent animate-grid-scan-vertical"></div>
      </div>

      {/* Central professional hub */}
      <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-32 h-32 animate-core-pulse">
        <div className="absolute inset-0 bg-gradient-to-br from-emerald-400/20 via-blue-500/20 to-violet-600/20 rounded-lg animate-spin-slow"></div>
        <div className="absolute inset-3 bg-gradient-to-br from-emerald-300/30 via-blue-400/30 to-violet-500/30 rounded-lg animate-spin-reverse"></div>
        <div className="absolute inset-6 bg-gradient-to-br from-emerald-200/40 via-blue-300/40 to-violet-400/40 rounded-lg animate-pulse"></div>
        <div className="absolute inset-9 bg-white/15 rounded-lg animate-ping"></div>
      </div>

      {/* Career symbols floating */}
      <div className="absolute top-16 left-1/2 transform -translate-x-1/2">
        <div className="flex space-x-6 animate-symbol-float">
          <div className="w-7 h-7 bg-gradient-to-br from-emerald-400/50 to-green-500/50 rounded-lg flex items-center justify-center text-white font-bold animate-bounce text-sm">
            💼
          </div>
          <div className="w-7 h-7 bg-gradient-to-br from-blue-400/50 to-cyan-500/50 rounded-lg flex items-center justify-center text-white font-bold animate-bounce text-sm" style={{animationDelay: '0.5s'}}>
            🎯
          </div>
          <div className="w-7 h-7 bg-gradient-to-br from-violet-400/50 to-purple-500/50 rounded-lg flex items-center justify-center text-white font-bold animate-bounce text-sm" style={{animationDelay: '1s'}}>
            🚀
          </div>
        </div>
      </div>

      {/* Professional wave animation */}
      <div className="absolute bottom-0 left-0 w-full h-28 overflow-hidden">
        <div className="absolute bottom-0 left-0 w-full h-full bg-gradient-to-t from-slate-800/60 to-transparent animate-wave"></div>
        <div className="absolute bottom-0 left-0 w-full h-full bg-gradient-to-t from-emerald-900/20 to-transparent animate-wave-reverse"></div>
      </div>

      {/* Subtle career ladder visualization */}
      <div className="absolute left-8 top-1/4 h-48 w-1 bg-gradient-to-b from-emerald-400/30 via-blue-400/30 to-violet-400/30 animate-pulse">
        <div className="absolute -left-2 top-0 w-5 h-1 bg-emerald-400/50 rounded-full"></div>
        <div className="absolute -left-2 top-16 w-5 h-1 bg-blue-400/50 rounded-full"></div>
        <div className="absolute -left-2 top-32 w-5 h-1 bg-violet-400/50 rounded-full"></div>
        <div className="absolute -left-2 top-48 w-5 h-1 bg-amber-400/50 rounded-full"></div>
      </div>
    </div>
  );
};

export default OccupationBackground;
