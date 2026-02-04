import { Routes, Route } from 'react-router-dom'
import LandingPage from './pages/LandingPage'
import ChatPage from './pages/ChatPage'
import SignInPage from './pages/SignInPage'
import SignUpPage from './pages/SignUpPage'

function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/sign-in/*" element={<SignInPage />} />
      <Route path="/sign-up/*" element={<SignUpPage />} />
      <Route path="/chat" element={<ChatPage />} />
    </Routes>
  )
}

export default App
