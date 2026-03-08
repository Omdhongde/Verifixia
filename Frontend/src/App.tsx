import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { useEffect, useState } from "react";
import { AppLayout } from "@/components/layout/AppLayout";
import Dashboard from "./pages/Dashboard";
import Login from "./pages/Login";
import Profile from "./pages/Profile";
import ForensicLogs from "./pages/ForensicLogs";
import Analytics from "./pages/Analytics";
import Settings from "./pages/Settings";
import Support from "./pages/Support";
import NotFound from "./pages/NotFound";
import { watchAuthState, firebaseEnabled } from "@/lib/auth";

const queryClient = new QueryClient();

// ── Shared auth state hook ────────────────────────────────────────────────────
function useAuth() {
  const [isReady, setIsReady] = useState(!firebaseEnabled); // instantly ready if Firebase off
  const [isAuthed, setIsAuthed] = useState(false);

  useEffect(() => {
    const unsub = watchAuthState((user) => {
      setIsAuthed(!!user);
      setIsReady(true);
    });
    return () => unsub();
  }, []);

  return { isReady, isAuthed };
}

// ── Loading spinner shown while Firebase resolves auth state ─────────────────
const AuthLoadingScreen = () => (
  <div className="min-h-screen bg-background flex items-center justify-center">
    <div className="flex flex-col items-center gap-4">
      <div className="w-10 h-10 border-4 border-primary border-t-transparent rounded-full animate-spin" />
      <p className="text-muted-foreground text-sm">Verifying session…</p>
    </div>
  </div>
);

// ── Guards ────────────────────────────────────────────────────────────────────

/** Redirect unauthenticated visitors to /login */
const ProtectedRoute = ({ children }: { children: JSX.Element }) => {
  const { isReady, isAuthed } = useAuth();
  if (!isReady) return <AuthLoadingScreen />;
  return isAuthed ? children : <Navigate to="/login" replace />;
};

/** Redirect already-logged-in users away from /login → home */
const PublicOnlyRoute = ({ children }: { children: JSX.Element }) => {
  const { isReady, isAuthed } = useAuth();
  if (!isReady) return <AuthLoadingScreen />;
  return isAuthed ? <Navigate to="/" replace /> : children;
};

// ── App ───────────────────────────────────────────────────────────────────────
const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Routes>
          {/* Public – only accessible when NOT logged in */}
          <Route path="/login" element={<PublicOnlyRoute><Login /></PublicOnlyRoute>} />

          {/* Protected – redirect to /login when not logged in */}
          <Route element={<ProtectedRoute><AppLayout /></ProtectedRoute>}>
            <Route path="/" element={<Dashboard />} />
            <Route path="/forensic-logs" element={<ForensicLogs />} />
            <Route path="/analytics" element={<Analytics />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="/support" element={<Support />} />
            <Route path="/profile" element={<Profile />} />
          </Route>

          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
