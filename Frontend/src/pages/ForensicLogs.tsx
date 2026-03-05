import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Search,
  Filter,
  Download,
  AlertTriangle,
  CheckCircle,
  Clock,
  Trash2,
  ChevronsLeft,
  ChevronsRight,
  FileDown,
} from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { clearDetectionLogs, deleteDetectionLog, fetchDetectionLogs } from "../../api";

type LogStatus = "threat" | "safe" | "review";

interface BackendLog {
  id?: string;
  timestamp: string;
  filename: string;
  prediction: string;
  confidence: number;
  session_id?: string;
  source_type?: string;
  model_version?: string;
  latency_ms?: number;
}

interface LogEntry {
  id: string;
  timestamp: string;
  timestampRaw: string;
  sessionId: string;
  source: string;
  sourceType: string;
  confidenceScore: number;
  status: LogStatus;
  actionTaken: string;
  modelVersion: string;
  latencyMs: number;
}

const getStatusBadge = (status: LogEntry["status"]) => {
  switch (status) {
    case "threat":
      return (
        <Badge className="bg-destructive/20 text-destructive border-destructive/30">
          <AlertTriangle className="w-3 h-3 mr-1" />
          Threat
        </Badge>
      );
    case "safe":
      return (
        <Badge className="bg-success/20 text-success border-success/30">
          <CheckCircle className="w-3 h-3 mr-1" />
          Safe
        </Badge>
      );
    default:
      return (
        <Badge className="bg-warning/20 text-warning border-warning/30">
          <Clock className="w-3 h-3 mr-1" />
          Review
        </Badge>
      );
  }
};

const getConfidenceColor = (score: number) => {
  if (score >= 70) return "text-destructive";
  if (score >= 40) return "text-warning";
  return "text-success";
};

function mapBackendToUiLogs(logs: BackendLog[]): LogEntry[] {
  return logs.map((log, index) => {
    const confidenceScore =
      typeof log.confidence === "number"
        ? Math.round(log.confidence <= 1 ? log.confidence * 100 : log.confidence)
        : 0;

    let status: LogStatus = "review";
    if ((log.prediction || "").toLowerCase() === "fake") {
      status = confidenceScore >= 65 ? "threat" : "review";
    } else if ((log.prediction || "").toLowerCase() === "real") {
      status = "safe";
    }

    let actionTaken = "Pending Review";
    if (status === "threat") actionTaken = "Flagged & Quarantined";
    if (status === "safe") actionTaken = "Verified Authentic";

    const parsedTs = new Date(log.timestamp);
    const timestamp = Number.isNaN(parsedTs.getTime()) ? log.timestamp : parsedTs.toLocaleString();

    return {
      id: log.id || `row-${index}-${log.timestamp}`,
      timestamp,
      timestampRaw: log.timestamp,
      sessionId: log.session_id || "N/A",
      source: log.filename || "Forensic Event",
      sourceType: log.source_type || "upload",
      confidenceScore,
      status,
      actionTaken,
      modelVersion: log.model_version || "N/A",
      latencyMs: Number(log.latency_ms || 0),
    };
  });
}

export const ForensicLogs = () => {
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [sourceFilter, setSourceFilter] = useState<string>("all");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);
  const [loading, setLoading] = useState(true);
  const [total, setTotal] = useState(0);
  const [rawLogs, setRawLogs] = useState<BackendLog[]>([]);

  const loadLogs = useCallback(async () => {
    setLoading(true);
    try {
      const data = await fetchDetectionLogs({
        page,
        page_size: pageSize,
        start_date: startDate ? `${startDate}T00:00:00` : "",
        end_date: endDate ? `${endDate}T23:59:59` : "",
        source_type: sourceFilter === "all" ? "" : sourceFilter,
      });
      const items = Array.isArray(data?.items) ? data.items : [];
      setRawLogs(items);
      setTotal(Number(data?.total || items.length));
    } catch (error) {
      console.error(error);
      toast.error("Failed to load forensic logs");
    } finally {
      setLoading(false);
    }
  }, [page, pageSize, startDate, endDate, sourceFilter]);

  useEffect(() => {
    loadLogs();
  }, [loadLogs]);

  const uiLogs = useMemo(() => mapBackendToUiLogs(rawLogs), [rawLogs]);
  const totalPages = Math.max(1, Math.ceil(total / pageSize));

  const filteredLogs = uiLogs.filter((log) => {
    const q = searchQuery.toLowerCase();
    const matchesSearch =
      log.id.toLowerCase().includes(q) ||
      log.sessionId.toLowerCase().includes(q) ||
      log.source.toLowerCase().includes(q) ||
      log.modelVersion.toLowerCase().includes(q);
    const matchesStatus = statusFilter === "all" || log.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  const handleDeleteLog = async (logId: string) => {
    try {
      await deleteDetectionLog(logId);
      toast.success("Log deleted");
      loadLogs();
    } catch (error) {
      console.error(error);
      toast.error("Could not delete this log");
    }
  };

  const handleClearLogs = async () => {
    try {
      await clearDetectionLogs(sourceFilter === "all" ? "" : sourceFilter);
      toast.success("Logs cleared");
      setPage(1);
      loadLogs();
    } catch (error) {
      console.error(error);
      toast.error("Could not clear logs");
    }
  };

  const handleExportCSV = () => {
    if (filteredLogs.length === 0) {
      toast.info("No logs to export");
      return;
    }
    const headers = ["ID", "Timestamp", "Source", "Source Type", "Session ID", "Model Version", "Latency (ms)", "Confidence (%)", "Status", "Action Taken"];
    const rows = filteredLogs.map((log) => [
      log.id,
      log.timestamp,
      log.source,
      log.sourceType,
      log.sessionId,
      log.modelVersion,
      log.latencyMs,
      log.confidenceScore,
      log.status,
      log.actionTaken,
    ]);
    const csv = [headers, ...rows]
      .map((row) => row.map((v) => `"${String(v ?? "").replace(/"/g, '""')}"`).join(","))
      .join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `verifixia_logs_${new Date().toISOString().slice(0, 10)}.csv`;
    link.click();
    URL.revokeObjectURL(url);
    toast.success(`Exported ${filteredLogs.length} log(s) to CSV`);
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Forensic Logs</h1>
        <p className="text-muted-foreground">Paginated forensic history for uploads and live monitoring events.</p>
      </div>

      <Card className="glass-card border-border/50">
        <CardHeader>
          <div className="flex flex-col gap-3">
            <div className="flex flex-col sm:flex-row gap-3 sm:items-center sm:justify-between">
              <div className="relative flex-1 max-w-md">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input
                  placeholder="Search by ID, session, source, model..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10"
                />
              </div>
              <div className="flex items-center gap-2">
                <Button variant="outline" onClick={loadLogs} disabled={loading}>
                  <Download className="w-4 h-4 mr-2" />
                  Refresh
                </Button>
                <Button variant="outline" onClick={handleExportCSV} disabled={loading || filteredLogs.length === 0}>
                  <FileDown className="w-4 h-4 mr-2" />
                  Export CSV
                </Button>
                <Button variant="destructive" onClick={handleClearLogs} disabled={loading || total === 0}>
                  <Trash2 className="w-4 h-4 mr-2" />
                  Clear Logs
                </Button>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <Select value={statusFilter} onValueChange={setStatusFilter}>
                <SelectTrigger className="w-[150px]">
                  <Filter className="w-4 h-4 mr-2" />
                  <SelectValue placeholder="Status" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Status</SelectItem>
                  <SelectItem value="threat">Threat</SelectItem>
                  <SelectItem value="safe">Safe</SelectItem>
                  <SelectItem value="review">Review</SelectItem>
                </SelectContent>
              </Select>

              <Select value={sourceFilter} onValueChange={(value) => { setSourceFilter(value); setPage(1); }}>
                <SelectTrigger className="w-[170px]">
                  <SelectValue placeholder="Source Type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Sources</SelectItem>
                  <SelectItem value="upload">Uploads</SelectItem>
                  <SelectItem value="live">Live Events</SelectItem>
                </SelectContent>
              </Select>

              <Input
                type="date"
                value={startDate}
                onChange={(e) => { setStartDate(e.target.value); setPage(1); }}
                className="w-[170px]"
              />
              <Input
                type="date"
                value={endDate}
                onChange={(e) => { setEndDate(e.target.value); setPage(1); }}
                className="w-[170px]"
              />
              <Select value={String(pageSize)} onValueChange={(value) => { setPageSize(Number(value)); setPage(1); }}>
                <SelectTrigger className="w-[140px]">
                  <SelectValue placeholder="Page Size" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="10">10 / page</SelectItem>
                  <SelectItem value="20">20 / page</SelectItem>
                  <SelectItem value="50">50 / page</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </CardHeader>

        <CardContent>
          <div className="rounded-lg border border-border/50 overflow-hidden">
            <Table>
              <TableHeader>
                <TableRow className="bg-secondary/30 hover:bg-secondary/30">
                  <TableHead>ID</TableHead>
                  <TableHead>Timestamp</TableHead>
                  <TableHead>Source</TableHead>
                  <TableHead>Session</TableHead>
                  <TableHead>Model</TableHead>
                  <TableHead>Latency</TableHead>
                  <TableHead>Confidence</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Action</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredLogs.length === 0 && (
                  <TableRow>
                    <TableCell colSpan={9} className="text-center py-8 text-muted-foreground">
                      {loading ? "Loading logs..." : "No logs found for the selected filters."}
                    </TableCell>
                  </TableRow>
                )}

                {filteredLogs.map((log) => (
                  <TableRow key={log.id} className="hover:bg-secondary/20">
                    <TableCell className="font-mono text-xs">{log.id}</TableCell>
                    <TableCell className="text-muted-foreground">{log.timestamp}</TableCell>
                    <TableCell>{log.source} <span className="text-xs text-muted-foreground">({log.sourceType})</span></TableCell>
                    <TableCell className="font-mono text-xs">{log.sessionId}</TableCell>
                    <TableCell>{log.modelVersion}</TableCell>
                    <TableCell className="font-mono">{Math.round(log.latencyMs)} ms</TableCell>
                    <TableCell>
                      <span className={`font-mono font-bold ${getConfidenceColor(log.confidenceScore)}`}>
                        {log.confidenceScore}%
                      </span>
                    </TableCell>
                    <TableCell>{getStatusBadge(log.status)}</TableCell>
                    <TableCell>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleDeleteLog(log.id)}
                        disabled={loading}
                      >
                        <Trash2 className="w-4 h-4 mr-1" />
                        Delete
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>

          <div className="mt-4 flex items-center justify-between">
            <p className="text-sm text-muted-foreground">
              Page {page} of {totalPages} | Total logs: {total}
            </p>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page <= 1 || loading}
              >
                <ChevronsLeft className="w-4 h-4 mr-1" />
                Prev
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                disabled={page >= totalPages || loading}
              >
                Next
                <ChevronsRight className="w-4 h-4 ml-1" />
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default ForensicLogs;
