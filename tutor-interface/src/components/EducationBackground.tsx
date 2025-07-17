
import React, { useState, useEffect, useRef } from 'react';

const EducationBackground = () => {
  const [particles, setParticles] = useState([]);
  const [dataStreams, setDataStreams] = useState([]);
  const containerRef = useRef(null);

  // Generate floating particles
  useEffect(() => {
    const generateParticles = () => {
      const newParticles = [];
      for (let i = 0; i < 60; i++) {
        newParticles.push({
          id: i,
          x: Math.random() * 100,
          y: Math.random() * 100,
          size: Math.random() * 8 + 2,
          opacity: Math.random() * 0.8 + 0.2,
          speed: Math.random() * 0.15 + 0.05,
          direction: Math.random() * Math.PI * 2,
          color: ['cyan', 'purple', 'blue', 'indigo', 'pink'][Math.floor(Math.random() * 5)],
          pulse: Math.random() * 2 + 1
        });
      }
      setParticles(newParticles);
    };

    const generateDataStreams = () => {
      const streams = [];
      for (let i = 0; i < 8; i++) {
        streams.push({
          id: i,
          x: Math.random() * 100,
          y: Math.random() * 100,
          width: Math.random() * 200 + 100,
          height: Math.random() * 4 + 2,
          angle: Math.random() * 360,
          speed: Math.random() * 2 + 1,
          color: ['cyan', 'purple', 'blue'][Math.floor(Math.random() * 3)]
        });
      }
      setDataStreams(streams);
    };

    generateParticles();
    generateDataStreams();

    // Animate particles
    const interval = setInterval(() => {
      setParticles(prev => prev.map(p => ({
        ...p,
        x: (p.x + Math.cos(p.direction) * p.speed) % 100,
        y: (p.y + Math.sin(p.direction) * p.speed) % 100,
        direction: p.direction + 0.01
      })));

      setDataStreams(prev => prev.map(s => ({
        ...s,
        x: (s.x + s.speed * 0.1) % 100,
        angle: s.angle + 0.5
      })));
    }, 50);

    return () => clearInterval(interval);
  }, []);

  return (
    <div ref={containerRef} className="absolute inset-0 w-full h-full bg-gradient-to-br from-slate-900 via-indigo-900 to-purple-900 overflow-hidden">
      {/* Animated gradient overlay */}
      <div className="absolute inset-0 bg-gradient-to-br from-cyan-900/20 via-purple-900/30 to-indigo-900/20 animate-gradient-shift"></div>
      
      {/* Floating geometric shapes */}
      <div className="absolute top-10 left-10 w-20 h-20 border-4 border-cyan-400/30 rounded-lg animate-float-rotate"></div>
      <div className="absolute top-32 right-20 w-16 h-16 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-full animate-pulse-scale"></div>
      <div className="absolute bottom-40 left-16 w-24 h-24 border-4 border-indigo-400/30 rounded-full animate-float-reverse"></div>
      <div className="absolute bottom-20 right-32 w-18 h-18 bg-gradient-to-br from-cyan-500/20 to-blue-500/20 transform rotate-45 animate-spin-slow"></div>
      
      {/* Neural network pattern */}
      <div className="absolute top-1/4 left-1/4 w-32 h-32 animate-neural-pulse">
        <div className="absolute inset-0 border-2 border-cyan-300/20 rounded-full"></div>
        <div className="absolute inset-4 border-2 border-purple-300/30 rounded-full animate-ping"></div>
        <div className="absolute inset-8 border-2 border-blue-300/40 rounded-full animate-pulse"></div>
        <div className="absolute top-1/2 left-0 w-full h-0.5 bg-gradient-to-r from-transparent via-cyan-400/50 to-transparent animate-data-flow"></div>
        <div className="absolute left-1/2 top-0 w-0.5 h-full bg-gradient-to-b from-transparent via-purple-400/50 to-transparent animate-data-flow-vertical"></div>
      </div>

      {/* Knowledge orbs */}
      <div className="absolute top-20 right-1/4 w-12 h-12 bg-gradient-to-br from-yellow-400/30 to-orange-500/30 rounded-full animate-orbit-1 border-2 border-yellow-300/50">
        <div className="absolute inset-2 bg-white/20 rounded-full animate-pulse"></div>
      </div>
      <div className="absolute bottom-32 left-1/3 w-10 h-10 bg-gradient-to-br from-green-400/30 to-emerald-500/30 rounded-full animate-orbit-2 border-2 border-green-300/50">
        <div className="absolute inset-2 bg-white/20 rounded-full animate-pulse"></div>
      </div>
      <div className="absolute top-1/2 right-16 w-14 h-14 bg-gradient-to-br from-pink-400/30 to-rose-500/30 rounded-full animate-orbit-3 border-2 border-pink-300/50">
        <div className="absolute inset-2 bg-white/20 rounded-full animate-pulse"></div>
      </div>

      {/* Dynamic particles */}
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
            boxShadow: `0 0 ${particle.size * 2}px var(--${particle.color}-400)`,
            animationDuration: `${particle.pulse}s`
          }}
        />
      ))}

      {/* Data streams */}
      {dataStreams.map(stream => (
        <div
          key={stream.id}
          className="absolute animate-stream-flow"
          style={{
            left: `${stream.x}%`,
            top: `${stream.y}%`,
            width: `${stream.width}px`,
            height: `${stream.height}px`,
            background: `linear-gradient(90deg, transparent, var(--${stream.color}-400), transparent)`,
            transform: `rotate(${stream.angle}deg)`,
            opacity: 0.6
          }}
        >
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent animate-pulse"></div>
        </div>
      ))}

      {/* Holographic grid */}
      <div className="absolute inset-0 opacity-10">
        <div className="w-full h-full bg-gradient-to-r from-transparent via-cyan-500/20 to-transparent animate-grid-scan"></div>
        <div className="absolute inset-0 w-full h-full bg-gradient-to-b from-transparent via-purple-500/20 to-transparent animate-grid-scan-vertical"></div>
      </div>

      {/* Central energy core */}
      <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-40 h-40 animate-core-pulse">
        <div className="absolute inset-0 bg-gradient-to-br from-cyan-400/20 via-purple-500/20 to-indigo-600/20 rounded-full animate-spin-slow"></div>
        <div className="absolute inset-4 bg-gradient-to-br from-cyan-300/30 via-purple-400/30 to-indigo-500/30 rounded-full animate-spin-reverse"></div>
        <div className="absolute inset-8 bg-gradient-to-br from-cyan-200/40 via-purple-300/40 to-indigo-400/40 rounded-full animate-pulse"></div>
        <div className="absolute inset-12 bg-white/10 rounded-full animate-ping"></div>
      </div>

      {/* Educational symbols floating */}
      <div className="absolute top-16 left-1/2 transform -translate-x-1/2">
        <div className="flex space-x-8 animate-symbol-float">
          <div className="w-8 h-8 bg-gradient-to-br from-yellow-400/40 to-orange-500/40 rounded-lg flex items-center justify-center text-white font-bold animate-bounce">
            π
          </div>
          <div className="w-8 h-8 bg-gradient-to-br from-green-400/40 to-emerald-500/40 rounded-lg flex items-center justify-center text-white font-bold animate-bounce" style={{animationDelay: '0.5s'}}>
            ∑
          </div>
          <div className="w-8 h-8 bg-gradient-to-br from-blue-400/40 to-cyan-500/40 rounded-lg flex items-center justify-center text-white font-bold animate-bounce" style={{animationDelay: '1s'}}>
            ∞
          </div>
        </div>
      </div>

      {/* Bottom wave animation */}
      <div className="absolute bottom-0 left-0 w-full h-32 overflow-hidden">
        <div className="absolute bottom-0 left-0 w-full h-full bg-gradient-to-t from-indigo-900/50 to-transparent animate-wave"></div>
        <div className="absolute bottom-0 left-0 w-full h-full bg-gradient-to-t from-purple-900/30 to-transparent animate-wave-reverse"></div>
      </div>
    </div>
  );
};

export default EducationBackground;
