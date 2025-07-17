
import React, { useState } from 'react';
import EducationBackground from '../components/EducationBackground';
import AuthHeader from '../components/AuthHeader';
import AuthToggle from '../components/AuthToggle';
import AuthForm from '../components/AuthForm';

export default function AuthPage() {
  const [isLogin, setIsLogin] = useState(true);

  return (
    <div className="min-h-screen relative overflow-hidden">
      <EducationBackground />

      <div className="relative z-20 min-h-screen flex items-center justify-center p-4">
        <div className="w-full max-w-md">
          <AuthHeader isLogin={isLogin} />
          <AuthToggle isLogin={isLogin} setIsLogin={setIsLogin} />
          <AuthForm isLogin={isLogin} />
        </div>
      </div>
    </div>
  );
}
