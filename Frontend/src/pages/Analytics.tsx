import { useEffect, useState, useCallback } from "react";
import {
  BarChart3,
  TrendingUp,
  TrendingDown,
  Activity,
  Shield,
  AlertTriangle,
  CheckCircle,
  Clock,
  RefreshCw,
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import { fetchStats } from "../../api";

interface Stats {
  total_scans: number;
  threats_detected: number;
  safe_detections: number;
  avg_confidence: number;
  avg_latency_ms: number;
  upload_count: number;
  live_count: number;
  detection_trend: { name: string; threats: number; safe: number }[];
  threat_types: { type: string; count: number; percentage: number }[];
  source_distribution: { name: string; value: number }[];
}

const PIE_COLORS = [
  "hsl(var(--primary))",
  "hsl(var(--success))",
  "hsl(var(--warning))",
  "hsl(var(--muted-foreground))",
];

const EMPTY_STATS: Stats = {
  total_scans: 0,
  threats_detected: 0,
  safe_detections: 0,
  avg_confidence: 0,
  avg_latency_ms: 0,
  upload_count: 0,
  live_count: 0,
  detection_trend: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"].map((n) => ({ name: n, threats: 0, safe: 0 })),
  threat_types: [
    { type: "Face Swap", count: 0, percentage: 0 },
    { type: "Lip Sync", count: 0, percentage: 0 },
    { type: "Audio Clone", count: 0, percentage: 0 },
    { type: "Full Synthesis", count: 0, percentage: 0 },
  ],
  source_distribution: [
    { name: "Live Streams", value: 0 },
    { name: "File Uploads", value: 0 },
    { name: "API Calls", value: 0 },
  ],
};

export const Analytics = () => {
  const [stats, setStats] = useState<Stats>(EMPTY_STATS);
  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState<string>("");

  const loadStats = useCallback(async () => {
    setLoading(true);
    try {
      const data = await fetchStats();
      if (data) {
        setStats(data);
        setLastUpdated(new Date().toLocaleTimeString());
      }
    } catch (err) {
      console.error("Failed to load stats", err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadStats();
  }, [loadStats]);

  const threatRate =
    stats.total_scans > 0
      ? Math.round((stats.threats_detected / stats.total_scans) * 1000) / 10
      : 0;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Analytics Dashboard</h1>
          <p className="text-muted-foreground">
            Live insights from your real detection history
            {lastUpdated && (
              <span className="ml-2 text-xs font-mono text-muted-foreground/60">
                — refreshed {lastUpdated}
              </span>
            )}
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={loadStats} disabled={loading}>
          <RefreshCw className={`w-4 h-4 mr-2 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </Button>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="glass-card border-border/50">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Total Scans</p>
                <p className="text-3xl font-bold">{stats.total_scans.toLocaleString()}</p>
              </div>
              <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center">
                <Activity className="w-6 h-6 text-primary" />
              </div>
            </div>
            <div className="flex items-center gap-1 mt-2 text-sm">
              <BarChart3 className="w-4 h-4 text-muted-foreground" />
              <span className="text-muted-foreground">
                {stats.upload_count} uploads · {stats.live_count} live
              </span>
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card border-border/50">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Threats Detected</p>
                <p className="text-3xl font-bold text-destructive">
                  {stats.threats_detected.toLocaleString()}
                </p>
              </div>
              <div className="w-12 h-12 rounded-lg bg-destructive/10 flex items-center justify-center">
                <AlertTriangle className="w-6 h-6 text-destructive" />
              </div>
            </div>
            <div className="flex items-center gap-1 mt-2 text-sm">
              {threatRate > 0 ? (
                <TrendingUp className="w-4 h-4 text-destructive" />
              ) : (
                <TrendingDown className="w-4 h-4 text-success" />
              )}
              <span className={threatRate > 0 ? "text-destructive" : "text-success"}>
                {threatRate}%
              </span>
              <span className="text-muted-foreground"> threat rate</span>
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card border-border/50">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Safe Detections</p>
                <p className="text-3xl font-bold text-success">
                  {stats.safe_detections.toLocaleString()}
                </p>
              </div>
              <div className="w-12 h-12 rounded-lg bg-success/10 flex items-center justify-center">
                <CheckCircle className="w-6 h-6 text-success" />
              </div>
            </div>
            <div className="flex items-center gap-1 mt-2 text-sm">
              <Shield className="w-4 h-4 text-success" />
              <span className="text-success">{stats.avg_confidence}%</span>
              <span className="text-muted-foreground"> avg confidence</span>
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card border-border/50">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Avg Response</p>
                <p className="text-3xl font-bold">
                  {stats.avg_latency_ms > 0 ? `${stats.avg_latency_ms}ms` : "—"}
                </p>
              </div>
              <div className="w-12 h-12 rounded-lg bg-warning/10 flex items-center justify-center">
                <Clock className="w-6 h-6 text-warning" />
              </div>
            </div>
            <div className="flex items-center gap-1 mt-2 text-sm">
              <span className="text-muted-foreground">
                {stats.avg_latency_ms > 0 ? "model inference latency" : "no timing data yet"}
              </span>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Charts Row 1 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="glass-card border-border/50">
          <CardHeader>
            <CardTitle>Detection Trend</CardTitle>
            <CardDescription>Threat vs. safe detections — last 7 days</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={stats.detection_trend}>
                  <defs>
                    <linearGradient id="safeGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="hsl(var(--success))" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="hsl(var(--success))" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="threatGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="hsl(var(--destructive))" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="hsl(var(--destructive))" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis dataKey="name" stroke="hsl(var(--muted-foreground))" fontSize={12} />
                  <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} allowDecimals={false} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "hsl(var(--card))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: "8px",
                    }}
                  />
                  <Area type="monotone" dataKey="safe" stroke="hsl(var(--success))" fill="url(#safeGradient)" strokeWidth={2} name="Safe" />
                  <Area type="monotone" dataKey="threats" stroke="hsl(var(--destructive))" fill="url(#threatGradient)" strokeWidth={2} name="Threats" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card border-border/50">
          <CardHeader>
            <CardTitle>Source Distribution</CardTitle>
            <CardDescription>Requests broken down by source type</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[300px] flex items-center gap-6">
              <div className="flex-1">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie data={stats.source_distribution} cx="50%" cy="50%" innerRadius={60} outerRadius={100} paddingAngle={4} dataKey="value">
                      {stats.source_distribution.map((_, i) => (
                        <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "hsl(var(--card))",
                        border: "1px solid hsl(var(--border))",
                        borderRadius: "8px",
                      }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="space-y-3 min-w-[140px]">
                {stats.source_distribution.map((item, i) => {
                  const total = stats.source_distribution.reduce((s, x) => s + x.value, 0);
                  const pct = total > 0 ? Math.round((item.value / total) * 100) : 0;
                  return (
                    <div key={item.name} className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full flex-shrink-0" style={{ backgroundColor: PIE_COLORS[i % PIE_COLORS.length] }} />
                      <span className="text-sm">{item.name}</span>
                      <span className="text-sm font-mono text-muted-foreground ml-auto">{item.value} ({pct}%)</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Charts Row 2 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="glass-card border-border/50">
          <CardHeader>
            <CardTitle>Weekly Bar Comparison</CardTitle>
            <CardDescription>Daily threats vs. safe detections</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={stats.detection_trend} barCategoryGap="30%">
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis dataKey="name" stroke="hsl(var(--muted-foreground))" fontSize={12} />
                  <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} allowDecimals={false} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "hsl(var(--card))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: "8px",
                    }}
                  />
                  <Bar dataKey="safe" fill="hsl(var(--success))" name="Safe" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="threats" fill="hsl(var(--destructive))" name="Threats" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card border-border/50">
          <CardHeader>
            <CardTitle>Threat Categories</CardTitle>
            <CardDescription>Estimated breakdown by deepfake technique</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              {stats.threat_types.map((threat) => (
                <div key={threat.type} className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span>{threat.type}</span>
                    <span className="font-mono text-muted-foreground">
                      {threat.count} ({threat.percentage}%)
                    </span>
                  </div>
                  <Progress value={threat.percentage} className="h-2" />
                </div>
              ))}
              {stats.threats_detected === 0 && (
                <p className="text-sm text-muted-foreground text-center py-4">
                  No threats detected yet. Upload or monitor media to populate this chart.
                </p>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Summary Badges */}
      <Card className="glass-card border-border/50">
        <CardHeader>
          <CardTitle>Quick Summary</CardTitle>
          <CardDescription>At-a-glance system health based on real data</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-3">
            <Badge variant="outline" className="px-4 py-2 text-sm">
              <Activity className="w-4 h-4 mr-2" />
              {stats.total_scans} total scans
            </Badge>
            <Badge variant="outline" className="px-4 py-2 text-sm border-destructive/50 text-destructive">
              <AlertTriangle className="w-4 h-4 mr-2" />
              {stats.threats_detected} deepfakes caught
            </Badge>
            <Badge variant="outline" className="px-4 py-2 text-sm border-success/50 text-success">
              <CheckCircle className="w-4 h-4 mr-2" />
              {stats.safe_detections} verified authentic
            </Badge>
            <Badge variant="outline" className="px-4 py-2 text-sm">
              <Clock className="w-4 h-4 mr-2" />
              {stats.avg_latency_ms > 0 ? `${stats.avg_latency_ms}ms avg latency` : "No latency data yet"}
            </Badge>
            <Badge variant="outline" className="px-4 py-2 text-sm">
              <Shield className="w-4 h-4 mr-2" />
              {stats.avg_confidence}% avg confidence
            </Badge>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default Analytics;
