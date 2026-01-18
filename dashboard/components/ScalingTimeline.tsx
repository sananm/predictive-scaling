'use client';

import { motion } from 'framer-motion';
import { Clock, ArrowUp, ArrowDown, Minus, CheckCircle, Loader, XCircle, AlertCircle } from 'lucide-react';
import { useScalingDecisions } from '@/hooks/useApi';
import { format } from 'date-fns';

const ActionIcon = ({ action }: { action: string }) => {
  switch (action) {
    case 'scale_up':
      return <ArrowUp className="w-5 h-5 text-emerald-400" />;
    case 'scale_down':
      return <ArrowDown className="w-5 h-5 text-blue-400" />;
    default:
      return <Minus className="w-5 h-5 text-zinc-400" />;
  }
};

const StatusIcon = ({ status }: { status: string }) => {
  switch (status) {
    case 'completed':
      return <CheckCircle className="w-4 h-4 text-emerald-400" />;
    case 'in_progress':
      return <Loader className="w-4 h-4 text-amber-400 animate-spin" />;
    case 'failed':
      return <XCircle className="w-4 h-4 text-red-400" />;
    case 'pending':
      return <AlertCircle className="w-4 h-4 text-orange-400" />;
    default:
      return <Clock className="w-4 h-4 text-zinc-400" />;
  }
};

const getActionColor = (action: string) => {
  switch (action) {
    case 'scale_up':
      return 'border-emerald-500/30 bg-emerald-500/5';
    case 'scale_down':
      return 'border-blue-500/30 bg-blue-500/5';
    default:
      return 'border-zinc-700 bg-zinc-800/50';
  }
};

const getConfidenceColor = (confidence: number) => {
  if (confidence >= 0.8) return 'text-emerald-400';
  if (confidence >= 0.6) return 'text-amber-400';
  return 'text-orange-400';
};

interface TimelineDecision {
  id: string;
  timestamp: string;
  action: string;
  from_replicas: number;
  to_replicas: number;
  reason: string;
  status: string;
  confidence_score: number;
}

const transformDecision = (apiDecision: any): TimelineDecision => {
  // API response has to_replicas/from_replicas OR target_instances/current_instances
  const fromReplicas = apiDecision.from_replicas ?? apiDecision.current_instances ?? 0;
  const toReplicas = apiDecision.to_replicas ?? apiDecision.target_instances ?? 0;

  // Determine action from strategy, action field, or replica counts
  let action = 'no_change';
  const strategy = apiDecision.strategy?.toLowerCase() || '';

  // First check strategy field (most reliable from backend)
  if (strategy.includes('scale_up') || strategy === 'preemptive_burst' || strategy === 'flash_sale') {
    action = 'scale_up';
  } else if (strategy.includes('scale_down') || strategy === 'maintenance' || strategy.includes('scale_in')) {
    action = 'scale_down';
  } else if (apiDecision.action && apiDecision.action !== 'no_change') {
    // Use explicit action field if present
    action = apiDecision.action;
  } else {
    // Fall back to comparing replica counts
    if (toReplicas > fromReplicas) {
      action = 'scale_up';
    } else if (toReplicas < fromReplicas) {
      action = 'scale_down';
    }
  }

  return {
    id: apiDecision.id,
    timestamp: apiDecision.created_at || apiDecision.timestamp,
    action,
    from_replicas: fromReplicas,
    to_replicas: toReplicas,
    reason: apiDecision.reasoning || apiDecision.reason || `Scaling from ${fromReplicas} to ${toReplicas} instances`,
    status: apiDecision.status,
    confidence_score: apiDecision.confidence_score ?? 0.8,
  };
};

const demoDecisions: TimelineDecision[] = [
  {
    id: 'demo-1',
    timestamp: new Date(Date.now() - 5 * 60 * 1000).toISOString(),
    action: 'scale_up',
    from_replicas: 3,
    to_replicas: 6,
    reason: 'Predicted traffic spike from Flash Sale event (2.5x multiplier)',
    status: 'completed',
    confidence_score: 0.92,
  },
  {
    id: 'demo-2',
    timestamp: new Date(Date.now() - 25 * 60 * 1000).toISOString(),
    action: 'scale_up',
    from_replicas: 2,
    to_replicas: 3,
    reason: 'CPU utilization exceeded 75% threshold',
    status: 'completed',
    confidence_score: 0.87,
  },
  {
    id: 'demo-3',
    timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
    action: 'scale_down',
    from_replicas: 5,
    to_replicas: 2,
    reason: 'Traffic returned to baseline after marketing campaign',
    status: 'completed',
    confidence_score: 0.85,
  },
  {
    id: 'demo-4',
    timestamp: new Date(Date.now() - 4 * 60 * 60 * 1000).toISOString(),
    action: 'scale_up',
    from_replicas: 2,
    to_replicas: 5,
    reason: 'ML model predicted 3x traffic increase for product launch',
    status: 'completed',
    confidence_score: 0.94,
  },
];

export default function ScalingTimeline() {
  // Poll every 1.5 seconds for faster updates after simulations
  const { data, loading, error } = useScalingDecisions(15, 1500);

  const apiDecisions = data.length > 0 ? data.map(transformDecision) : [];
  const decisions = apiDecisions.length > 0 ? apiDecisions : demoDecisions;

  if (loading && data.length === 0) {
    return (
      <div className="glass rounded-2xl p-8">
        <div className="animate-pulse space-y-4">
          <div className="h-8 bg-zinc-800 rounded w-1/3"></div>
          <div className="h-32 bg-zinc-800 rounded"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="glass rounded-2xl p-8">
        <div className="text-center text-red-400">
          <p className="text-sm">Failed to load scaling decisions</p>
        </div>
      </div>
    );
  }

  return (
    <motion.div
      className="glass rounded-2xl p-8"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.4 }}
    >
      <div className="flex items-center gap-3 mb-8">
        <div className="bg-zinc-800 rounded-xl p-3">
          <Clock className="w-6 h-6 text-blue-500" />
        </div>
        <div>
          <h2 className="text-2xl font-bold">Scaling Timeline</h2>
          <p className="text-sm text-zinc-500 mt-1">
            Recent scaling decisions and actions
          </p>
        </div>
      </div>

      {decisions.length === 0 ? (
        <div className="text-center py-12 text-zinc-500">
          <Clock className="w-16 h-16 mx-auto mb-4 opacity-50" />
          <p>No scaling decisions yet</p>
          <p className="text-sm text-zinc-600 mt-2">Decisions will appear here as the system scales</p>
        </div>
      ) : (
        <div className="relative">
          {/* Timeline line */}
          <div className="absolute left-6 top-0 bottom-0 w-0.5 bg-zinc-700" />

          {/* Timeline items */}
          <div className="space-y-6">
            {decisions.map((decision, index) => (
              <motion.div
                key={decision.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.05, duration: 0.3 }}
                className="relative pl-16"
              >
                {/* Timeline node */}
                <div className="absolute left-3 top-1/2 -translate-y-1/2">
                  <motion.div
                    className={`w-6 h-6 rounded-full border-2 flex items-center justify-center bg-zinc-900 ${getActionColor(decision.action)}`}
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: index * 0.05 + 0.2, type: 'spring' }}
                  >
                    <ActionIcon action={decision.action} />
                  </motion.div>

                  {decision.status === 'in_progress' && (
                    <motion.div
                      className="absolute inset-0 rounded-full bg-amber-400"
                      animate={{ scale: [1, 2, 2], opacity: [0.5, 0, 0] }}
                      transition={{ duration: 2, repeat: Infinity }}
                    />
                  )}
                </div>

                {/* Decision card */}
                <div className={`rounded-xl p-4 border ${getActionColor(decision.action)}`}>
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <div className="flex items-center gap-2 mb-1">
                        <h3 className="font-semibold capitalize">
                          {decision.action.replace('_', ' ')}
                        </h3>
                        <StatusIcon status={decision.status} />
                      </div>
                      <p className="text-xs text-zinc-500">
                        {format(new Date(decision.timestamp), 'MMM d, yyyy • HH:mm:ss')}
                      </p>
                    </div>
                    <div className="text-right">
                      <div className="text-sm font-mono">
                        <span className="text-zinc-500">{decision.from_replicas}</span>
                        <span className="mx-2 text-zinc-600">→</span>
                        <span className="text-white font-bold">{decision.to_replicas}</span>
                      </div>
                    </div>
                  </div>

                  <p className="text-sm text-zinc-400 mb-3">{decision.reason}</p>

                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-zinc-600">Confidence:</span>
                      <span className={`text-sm font-bold ${getConfidenceColor(decision.confidence_score)}`}>
                        {(decision.confidence_score * 100).toFixed(0)}%
                      </span>
                    </div>
                    <span
                      className={`text-xs px-2 py-1 rounded-full ${decision.status === 'completed'
                        ? 'bg-emerald-500/20 text-emerald-400'
                        : decision.status === 'in_progress'
                          ? 'bg-amber-500/20 text-amber-400'
                          : decision.status === 'failed'
                            ? 'bg-red-500/20 text-red-400'
                            : 'bg-zinc-700 text-zinc-400'
                        }`}
                    >
                      {decision.status}
                    </span>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      )}

      {/* Summary stats */}
      {decisions.length > 0 && (
        <motion.div
          className="mt-8 grid grid-cols-3 gap-4"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8 }}
        >
          <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-xl p-4 text-center">
            <div className="text-2xl font-bold text-emerald-400">
              {decisions.filter((d) => d.action === 'scale_up').length}
            </div>
            <div className="text-xs text-zinc-500 mt-1">Scale Ups</div>
          </div>
          <div className="bg-blue-500/10 border border-blue-500/20 rounded-xl p-4 text-center">
            <div className="text-2xl font-bold text-blue-400">
              {decisions.filter((d) => d.action === 'scale_down').length}
            </div>
            <div className="text-xs text-zinc-500 mt-1">Scale Downs</div>
          </div>
          <div className="bg-zinc-800 border border-zinc-700 rounded-xl p-4 text-center">
            <div className="text-2xl font-bold text-white">
              {decisions.filter((d) => d.status === 'completed').length}
            </div>
            <div className="text-xs text-zinc-500 mt-1">Completed</div>
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}
