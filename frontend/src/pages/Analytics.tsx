import { BarChart2, AlertCircle } from "lucide-react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  PieChart, Pie,
} from "recharts";
import { Card, CardHeader, CardTitle } from "../components/ui/Card";
import { PageHeader, EmptyState } from "../components/ui/Primitives";
import { useAnalyticsStore } from "../store";

export default function Analytics() {
  const { attempts, accepted, rejected, latencies } = useAnalyticsStore();

  const avgLatency =
    latencies.length > 0
      ? Math.round(latencies.reduce((a, b) => a + b, 0) / latencies.length)
      : 0;

  const acceptanceRate = attempts > 0 ? Math.round((accepted / attempts) * 100) : 0;

  const barData = [
    { name: "Accepted", value: accepted, fill: "#16A34A" },
    { name: "Rejected", value: rejected, fill: "#DC2626" },
  ];

  const pieData = [
    { name: "Accepted", value: accepted || 1, fill: "#16A34A" },
    { name: "Rejected", value: rejected, fill: "#DC2626" },
  ];

  const hasData = attempts > 0;

  return (
    <div className="p-6 max-w-4xl mx-auto space-y-5">
      <PageHeader
        title="Recognition Analytics"
        description="Session-level statistics derived from actual recognition events. Data resets on page reload."
        icon={<BarChart2 size={20} />}
        iconColor="bg-indigo-50 text-brand-primary"
      />

      {!hasData ? (
        <EmptyState
          icon={<BarChart2 size={28} />}
          title="No recognition data yet"
          description="Run a sign recognition session to populate this dashboard. Only real inference events are counted."
        />
      ) : (
        <>
          {/* Stat row */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { label: "Total Attempts",   value: attempts,        color: "text-brand-primary" },
              { label: "Accepted",          value: accepted,        color: "text-status-success" },
              { label: "Rejected",          value: rejected,        color: "text-status-error" },
              { label: "Avg Latency",       value: `${avgLatency}ms`, color: "text-brand-accent" },
            ].map((s) => (
              <Card key={s.label}>
                <p className="text-xs text-content-secondary mb-1">{s.label}</p>
                <p className={`text-2xl font-bold ${s.color}`}>{s.value}</p>
              </Card>
            ))}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {/* Bar chart */}
            <Card>
              <CardHeader><CardTitle>Accepted vs Rejected</CardTitle></CardHeader>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={barData} margin={{ top: 0, right: 8, bottom: 0, left: -20 }}>
                  <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} allowDecimals={false} />
                  <Tooltip
                    contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #E2E8F0" }}
                    cursor={{ fill: "rgba(0,0,0,0.04)" }}
                  />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                    {barData.map((d, i) => (
                      <Cell key={i} fill={d.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>

            {/* Acceptance rate donut */}
            <Card className="flex flex-col items-center justify-center">
              <CardHeader><CardTitle>Acceptance Rate</CardTitle></CardHeader>
              <div className="relative">
                <ResponsiveContainer width={160} height={160}>
                  <PieChart>
                    <Pie
                      data={pieData}
                      cx="50%"
                      cy="50%"
                      innerRadius={50}
                      outerRadius={70}
                      dataKey="value"
                      startAngle={90}
                      endAngle={-270}
                      strokeWidth={0}
                    >
                      {pieData.map((d, i) => (
                        <Cell key={i} fill={d.fill} />
                      ))}
                    </Pie>
                  </PieChart>
                </ResponsiveContainer>
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-2xl font-bold text-content-primary">{acceptanceRate}%</span>
                </div>
              </div>
            </Card>
          </div>
        </>
      )}

      <p className="text-xs text-content-secondary">
        <AlertCircle size={12} className="inline mr-1" aria-hidden="true" />
        Confidence scores represent the classifier's softmax output, not independently validated real-world accuracy.
      </p>
    </div>
  );
}
