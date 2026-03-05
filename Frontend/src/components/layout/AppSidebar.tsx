import { useState } from "react";
import {
  Monitor,
  FileText,
  BarChart3,
  Settings,
  HelpCircle,
  ChevronLeft,
  ChevronRight,
  Shield,
} from "lucide-react";
import { NavLink } from "@/components/NavLink";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const navItems = [
  { title: "Live Monitor", url: "/", icon: Monitor },
  { title: "Forensic Logs", url: "/forensic-logs", icon: FileText },
  { title: "Analytics", url: "/analytics", icon: BarChart3 },
  { title: "Security Settings", url: "/settings", icon: Settings },
  { title: "Support", url: "/support", icon: HelpCircle },
];

interface AppSidebarProps {
  collapsed: boolean;
  onToggle: () => void;
}

export const AppSidebar = ({ collapsed, onToggle }: AppSidebarProps) => {
  return (
    <aside
      className={cn(
        "fixed left-0 top-0 z-40 h-screen glass-card border-r border-border/50 transition-all duration-300",
        collapsed ? "w-16" : "w-64"
      )}
    >
      {/* Logo */}
      <div className="flex items-center gap-3 px-4 py-6 border-b border-border/50">
        <div className="relative flex-shrink-0">
          <Shield className="w-8 h-8 text-primary" />
          <div className="absolute -top-0.5 -right-0.5 w-2 h-2 bg-primary rounded-full animate-pulse-glow" />
        </div>
        {!collapsed && (
          <div className="overflow-hidden">
            <h1 className="text-lg font-semibold tracking-tight truncate">
              <span className="text-primary text-glow-cyan">Verifixia</span>
              <span className="text-foreground"> AI</span>
            </h1>
          </div>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-3 space-y-1">
        {navItems.map((item) => (
          <NavLink
            key={item.url}
            to={item.url}
            end={item.url === "/"}
            className={cn(
              "flex items-center gap-3 px-3 py-2.5 rounded-lg text-muted-foreground transition-all hover:bg-secondary/50 hover:text-foreground group",
              collapsed && "justify-center px-2"
            )}
            activeClassName="bg-primary/10 text-primary border border-primary/20"
          >
            <item.icon className="w-5 h-5 flex-shrink-0" />
            {!collapsed && (
              <span className="text-sm font-medium truncate">{item.title}</span>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Collapse Toggle */}
      <div className="absolute bottom-4 right-0 translate-x-1/2">
        <Button
          variant="outline"
          size="icon"
          onClick={onToggle}
          className="h-6 w-6 rounded-full border-border bg-background shadow-md hover:bg-secondary"
        >
          {collapsed ? (
            <ChevronRight className="h-3 w-3" />
          ) : (
            <ChevronLeft className="h-3 w-3" />
          )}
        </Button>
      </div>

      {/* Version */}
      {!collapsed && (
        <div className="p-4 border-t border-border/50">
          <p className="text-xs text-muted-foreground font-mono">v2.4.1</p>
        </div>
      )}
    </aside>
  );
};
