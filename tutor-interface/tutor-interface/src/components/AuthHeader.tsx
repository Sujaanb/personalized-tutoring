
import React from 'react';
import { Brain } from 'lucide-react';

interface AuthHeaderProps {
  isLogin: boolean;
}

export default function AuthHeader({ isLogin }: AuthHeaderProps) {
  return (
    <div className="text-center mb-8">
      <div className="flex items-center justify-center mb-6">
        <div className="bg-gradient-to-r from-cyan-500 via-purple-600 to-indigo-600 p-4 rounded-2xl shadow-2xl animate-pulse">
          <Brain className="w-12 h-12 text-white" />
        </div>
      </div>
      <h1 className="text-5xl font-bold text-white mb-4 bg-gradient-to-r from-cyan-200 via-purple-200 to-indigo-200 bg-clip-text text-transparent drop-shadow-lg">
        AI Tutor
      </h1>
      <p className="text-slate-200 text-lg font-medium drop-shadow-md">
        {isLogin 
          ? 'Welcome to the future of learning' 
          : 'Join the educational revolution'
        }
      </p>
    </div>
  );
}
