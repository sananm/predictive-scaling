'use client';

import { motion } from 'framer-motion';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts';
import { TrendingUp, Brain } from 'lucide-react';
import { usePredictions } from '@/hooks/useApi';
import { format } from 'date-fns';

const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="glass rounded-lg p-4 border border-zinc-700">
        <p className="text-sm text-zinc-400 mb-2">
          {format(new Date(payload[0].payload.timestamp), 'MMM d, HH:mm')}
        </p>
        <div className="space-y-1">
          <p className="text-sm">
            <span className="text-blue-400">Predicted Load: </span>
            <span className="font-bold">{payload[0].value.toFixed(1)}</span>
          </p>
          {payload[1] && (
            <p className="text-sm">
              <span className="text-zinc-400">High Confidence: </span>
              <span className="font-bold">{payload[1].value.toFixed(1)}</span>
            </p>
          )}
          {payload[2] && (
            <p className="text-sm">
              <span className="text-zinc-500">Low Confidence: </span>
              <span className="font-bold">{payload[2].value.toFixed(1)}</span>
            </p>
          )}
        </div>
      </div>
    );
  }
  return null;
};

// Transform API predictions to chart format
const transformPredictions = (apiPredictions: any[]) => {
  return apiPredictions.map((pred: any) => ({
    timestamp: pred.target_time || pred.timestamp,
    predicted_load: pred.p50 || pred.prediction_p50 || pred.predicted_load || 0,
    confidence_high: pred.p90 || pred.prediction_p90 || pred.confidence_high || 0,
    confidence_low: pred.p10 || pred.prediction_p10 || pred.confidence_low || 0,
  }));
};

// Generate demo data for display when no real data
const generateDemoData = () => {
  const now = new Date();
  const data = [];
  for (let i = 0; i < 12; i++) {
    const timestamp = new Date(now.getTime() + i * 5 * 60 * 1000);
    const baseLoad = 65 + Math.sin(i * 0.5) * 20 + Math.random() * 10;
    data.push({
      timestamp: timestamp.toISOString(),
      predicted_load: baseLoad,
      confidence_high: baseLoad + 15 + Math.random() * 5,
      confidence_low: baseLoad - 15 - Math.random() * 5,
    });
  }
  return data;
};

export default function PredictionsChart() {
  const { data, loading, error, recentTriggerResults } = usePredictions();

  // Use triggered predictions first, then API predictions, then demo data
  const apiPredictions = data?.predictions || [];
  const triggeredPredictions = recentTriggerResults || [];

  // Combine and transform predictions
  const allPredictions = triggeredPredictions.length > 0
    ? transformPredictions(triggeredPredictions)
    : apiPredictions.length > 0
      ? transformPredictions(apiPredictions)
      : generateDemoData();

  const isDemo = triggeredPredictions.length === 0 && apiPredictions.length === 0;
  const predictions = allPredictions;

  if (loading && !data) {
    return (
      <div className="glass rounded-2xl p-8">
        <div className="animate-pulse space-y-4">
          <div className="h-8 bg-zinc-800 rounded w-1/3"></div>
          <div className="h-80 bg-zinc-800 rounded"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="glass rounded-2xl p-8">
        <div className="text-center text-red-400">
          <p className="text-sm">Failed to load predictions</p>
          <p className="text-xs text-zinc-500 mt-2">Make sure the API is running and models are trained</p>
        </div>
      </div>
    );
  }

  const chartData = predictions.map((pred: any) => ({
    timestamp: pred.timestamp,
    predicted: pred.predicted_load,
    high: pred.confidence_high,
    low: pred.confidence_low,
  }));

  return (
    <motion.div
      className="glass rounded-2xl p-8"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.2 }}
    >
      <div className="flex items-center justify-between mb-8">
        <div className="flex items-center gap-3">
          <div className="bg-zinc-800 rounded-xl p-3">
            <TrendingUp className="w-6 h-6 text-blue-500" />
          </div>
          <div>
            <h2 className="text-2xl font-bold">Predictive Analytics</h2>
            <p className="text-sm text-zinc-500 mt-1">
              ML-powered load forecast with confidence intervals
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2 px-4 py-2 bg-zinc-800 rounded-full border border-zinc-700">
          <Brain className="w-4 h-4 text-blue-500" />
          <span className="text-sm text-zinc-400 font-medium">
            {isDemo ? 'Demo Mode' : `Model: ${data?.model_version || '1.0'}`}
          </span>
        </div>
      </div>

      {chartData.length === 0 ? (
        <div className="h-80 flex items-center justify-center text-zinc-500">
          <div className="text-center">
            <Brain className="w-16 h-16 mx-auto mb-4 opacity-50" />
            <p>No predictions available yet</p>
            <p className="text-sm text-zinc-600 mt-2">Trigger a prediction to see forecasts</p>
          </div>
        </div>
      ) : (
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="colorPredicted" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="colorHigh" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#52525b" stopOpacity={0.2} />
                  <stop offset="95%" stopColor="#52525b" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="colorLow" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3f3f46" stopOpacity={0.2} />
                  <stop offset="95%" stopColor="#3f3f46" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
              <XAxis
                dataKey="timestamp"
                tickFormatter={(value) => format(new Date(value), 'HH:mm')}
                stroke="#52525b"
                style={{ fontSize: '12px' }}
              />
              <YAxis
                stroke="#52525b"
                style={{ fontSize: '12px' }}
              />
              <Tooltip content={<CustomTooltip />} />
              <Area
                type="monotone"
                dataKey="high"
                stroke="#52525b"
                strokeWidth={1}
                fill="url(#colorHigh)"
                strokeDasharray="5 5"
              />
              <Area
                type="monotone"
                dataKey="low"
                stroke="#3f3f46"
                strokeWidth={1}
                fill="url(#colorLow)"
                strokeDasharray="5 5"
              />
              <Line
                type="monotone"
                dataKey="predicted"
                stroke="#3b82f6"
                strokeWidth={3}
                dot={false}
                fill="url(#colorPredicted)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      <motion.div
        className="mt-6 flex items-center justify-center gap-6 text-sm"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
      >
        <div className="flex items-center gap-2">
          <div className="w-3 h-0.5 bg-blue-500"></div>
          <span className="text-zinc-500">Predicted Load</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-0.5 bg-zinc-500 opacity-50"></div>
          <span className="text-zinc-500">P90 Confidence</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-0.5 bg-zinc-600 opacity-50"></div>
          <span className="text-zinc-500">P10 Confidence</span>
        </div>
      </motion.div>
    </motion.div>
  );
}
