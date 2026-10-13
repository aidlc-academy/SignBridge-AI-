import { useState, useMemo } from "react";
import { History as HistoryIcon, Trash2, Search, Download } from "lucide-react";
import { Card } from "../components/ui/Card";
import { Button } from "../components/ui/Button";
import { PageHeader, EmptyState } from "../components/ui/Primitives";
import { useConversationStore } from "../store";

export default function History() {
  const { conversations, setConversations } = useConversationStore();
  const [query, setQuery] = useState("");

  const filtered = useMemo(() => {
    if (!query.trim()) return conversations;
    const q = query.toLowerCase();
    return conversations.filter((c) =>
      c.messages.some((m) => m.content.toLowerCase().includes(q))
    );
  }, [conversations, query]);

  const deleteOne = (id: string) => {
    setConversations(conversations.filter((c) => c.id !== id));
  };

  const clearAll = () => {
    if (window.confirm("Delete all conversation history? This cannot be undone.")) {
      setConversations([]);
    }
  };

  const exportTxt = () => {
    const lines = conversations.flatMap((c) =>
      [`Session: ${c.id.slice(0, 8)} — ${new Date(c.createdAt).toLocaleString()}`,
       ...c.messages.map((m) => `  [${m.source}] ${m.content}`),
       ""]
    );
    const blob = new Blob([lines.join("\n")], { type: "text/plain" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href     = url;
    a.download = "signbridge-history.txt";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="p-6 max-w-4xl mx-auto space-y-5">
      <PageHeader
        title="Conversation History"
        description="Opt-in text-only history. No raw frames or audio are stored."
        icon={<HistoryIcon size={20} />}
        iconColor="bg-slate-100 text-content-secondary"
        actions={
          conversations.length > 0 ? (
            <div className="flex items-center gap-2">
              <Button variant="ghost" size="sm" icon={<Download size={14} />} onClick={exportTxt}>
                Export
              </Button>
              <Button variant="danger" size="sm" icon={<Trash2 size={14} />} onClick={clearAll}>
                Clear All
              </Button>
            </div>
          ) : undefined
        }
      />

      {/* Search */}
      {conversations.length > 0 && (
        <div className="relative">
          <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-content-secondary" aria-hidden="true" />
          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search messages…"
            className="w-full pl-9 pr-4 py-2.5 text-sm border border-surface-border rounded-xl bg-white focus:ring-2 focus:ring-brand-primary focus:outline-none"
            aria-label="Search conversation history"
          />
        </div>
      )}

      {conversations.length === 0 ? (
        <EmptyState
          icon={<HistoryIcon size={28} />}
          title="No saved conversations"
          description="After completing a sign or speech session, save it to view it here. History is opt-in."
        />
      ) : filtered.length === 0 ? (
        <EmptyState
          icon={<Search size={28} />}
          title="No matching conversations"
          description={`No messages contain "${query}".`}
        />
      ) : (
        <div className="space-y-3" role="list" aria-label="Conversation history">
          {filtered.map((c) => (
            <Card key={c.id} role="listitem">
              <div className="flex items-start justify-between gap-4">
                <div className="min-w-0">
                  <p className="text-sm font-medium text-content-primary">
                    Session <code className="text-xs bg-slate-100 px-1 rounded">{c.id.slice(0, 8)}</code>
                  </p>
                  <p className="text-xs text-content-secondary mt-0.5">
                    {new Date(c.createdAt).toLocaleString()} · {c.messages.length} message{c.messages.length !== 1 ? "s" : ""}
                  </p>
                  {c.messages.length > 0 && (
                    <p className="text-sm text-content-secondary mt-2 line-clamp-2">
                      {c.messages.slice(0, 2).map((m) => m.content).join(" · ")}
                    </p>
                  )}
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  icon={<Trash2 size={14} />}
                  onClick={() => deleteOne(c.id)}
                  aria-label={`Delete session ${c.id.slice(0, 8)}`}
                >
                  Delete
                </Button>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
