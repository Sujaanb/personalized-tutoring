
import React from 'react';

interface AuthToggleProps {
  isLogin: boolean;
  setIsLogin: (isLogin: boolean) => void;
}

export default function AuthToggle({ isLogin, setIsLogin }: AuthToggleProps) {
  return (
    <div className="flex bg-slate-800/40 backdrop-blur-sm rounded-2xl p-2 mb-8 border border-slate-700/50">
      <button
        onClick={() => setIsLogin(true)}
        className={`flex-1 py-3 px-6 rounded-xl font-semibold transition-all duration-300 ${
          isLogin
            ? 'bg-gradient-to-r from-cyan-600 to-indigo-600 text-white shadow-lg'
            : 'text-slate-300 hover:text-slate-100'
        }`}
      >
        Sign In
      </button>
      <button
        onClick={() => setIsLogin(false)}
        className={`flex-1 py-3 px-6 rounded-xl font-semibold transition-all duration-300 ${
          !isLogin
            ? 'bg-gradient-to-r from-cyan-600 to-indigo-600 text-white shadow-lg'
            : 'text-slate-300 hover:text-slate-100'
        }`}
      >
        Sign Up
      </button>
    </div>
  );
}
