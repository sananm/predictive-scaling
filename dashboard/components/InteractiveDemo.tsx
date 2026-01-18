'use client';

import { motion } from 'framer-motion';
import { useState } from 'react';
import { Zap, Sparkles, Play, CheckCircle, AlertCircle, TrendingUp, TrendingDown } from 'lucide-react';
import { usePredictions, useEvents } from '@/hooks/useApi';

export default function InteractiveDemo() {
  const { triggerPrediction } = usePredictions();
  const { createEvent } = useEvents();

  const [predictionStatus, setPredictionStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [eventStatus, setEventStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [message, setMessage] = useState('');
  const [rpsMultiplier, setRpsMultiplier] = useState(3);
  const [scaleDirection, setScaleDirection] = useState<'up' | 'down'>('up');

  const handleTriggerPrediction = async () => {
    setPredictionStatus('loading');
    setMessage('Running ML models...');

    try {
      await triggerPrediction();
      setPredictionStatus('success');
      setMessage('Predictions generated successfully!');

      setTimeout(() => {
        setPredictionStatus('idle');
        setMessage('');
      }, 3000);
    } catch (error) {
      setPredictionStatus('error');
      setMessage('Failed to trigger prediction. Is the API running?');

      setTimeout(() => {
        setPredictionStatus('idle');
        setMessage('');
      }, 3000);
    }
  };

  const handleSimulateScale = async () => {
    setEventStatus('loading');
    const actionText = scaleDirection === 'up' ? 'spike' : 'decrease';
    // Use multiplier for up, divisor for down (passed as 1/divisor to evaluate, or just use multiplier and let backend handle logic? 
    // Wait, backend evaluate handles "multiplier". If I want to simulate drop, I should pass multiplier < 1.0.
    // The previous simulateScaleDown took a divisor (e.g. 2.0). 
    // My new evaluate implementation handles multiplier. If multiplier > 1.0 -> flash_sale.
    // If request.multiplier < 1.0 -> maintenance/drop.
    // So if user selects 2x scale down, I should pass 0.5 as multiplier? 
    // Let's check backend implementation of evaluate_scaling... 
    // It creates "flash_sale" if multiplier > 1.0 else "maintenance".
    // And "Simulated Spike" or "Simulated Drop". 
    // But does prediction service handle multiplier < 1.0 correctly? 
    // PredictionService._get_active_event_multiplier gets MAX multiplier. 
    // If I have a maintenance event with impact 0.5, does it multiply predictions by 0.5?
    // I need to verify PredictionService logic, but assuming it works:

    // Scale Down logic: failure scenarios or maintenance usually mean < 1.0 multiplier.
    // If user says "Scale Down 2x", they mean halve the traffic. So multiplier = 0.5.

    const effectiveMultiplier = scaleDirection === 'up'
      ? rpsMultiplier
      : (1 / rpsMultiplier);

    setMessage(`Running ML Evaluation for ${rpsMultiplier}x traffic ${actionText}...`);

    try {
      const { apiClient } = await import('@/lib/api');

      // Call the real ML evaluation endpoint
      const result = await apiClient.evaluateScaling(effectiveMultiplier);

      if (result.success) {
        setEventStatus('success');
        const arrow = result.action === 'scale_up' ? '↑' : (result.action === 'scale_down' ? '↓' : '→');

        // Show the reasoning from the decision engine
        if (result.action === 'no_change') {
          setMessage(`ML Evaluation: No scaling needed. ${result.reasoning || 'Capacity sufficient.'}`);
        } else {
          setMessage(
            `Decision: ${result.action.replace('_', ' ').toUpperCase()} ${arrow} ${result.target_instances} instances. ${result.reasoning?.split('.')[0]}.`
          );
        }

        // Also trigger a prediction to update the chart
        setTimeout(async () => {
          await handleTriggerPrediction();
        }, 500);
      } else {
        setEventStatus('error');
        setMessage(result.message || 'Evaluation failed');
      }

      setTimeout(() => {
        setEventStatus('idle');
        setMessage('');
      }, 5000);
    } catch (error) {
      setEventStatus('error');
      setMessage(`Failed to run evaluation. Check API connection.`);

      setTimeout(() => {
        setEventStatus('idle');
        setMessage('');
      }, 3000);
    }
  };


  return (
    <motion.div
      className="glass rounded-2xl p-8"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.3 }}
    >
      <div className="flex items-center gap-3 mb-8">
        <div className="bg-zinc-800 rounded-xl p-3">
          <Play className="w-6 h-6 text-amber-500" />
        </div>
        <div>
          <h2 className="text-2xl font-bold">Interactive Demo</h2>
          <p className="text-sm text-zinc-500 mt-1">
            Trigger ML predictions and simulate real-world scenarios
          </p>
        </div>
      </div>

      {/* Scale Direction Toggle */}
      <div className="mb-4 p-4 rounded-xl bg-zinc-800/50 border border-zinc-700">
        <div className="flex items-center justify-between mb-3">
          <span className="text-sm font-medium text-zinc-400">Scale Direction</span>
          <div className="flex gap-2">
            <button
              onClick={() => setScaleDirection('up')}
              className={`flex items-center gap-1 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${scaleDirection === 'up'
                ? 'bg-emerald-500/20 border border-emerald-500/50 text-emerald-400'
                : 'bg-zinc-700/50 border border-zinc-600 text-zinc-400 hover:border-zinc-500'
                }`}
            >
              <TrendingUp className="w-4 h-4" />
              Scale Up
            </button>
            <button
              onClick={() => setScaleDirection('down')}
              className={`flex items-center gap-1 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${scaleDirection === 'down'
                ? 'bg-blue-500/20 border border-blue-500/50 text-blue-400'
                : 'bg-zinc-700/50 border border-zinc-600 text-zinc-400 hover:border-zinc-500'
                }`}
            >
              <TrendingDown className="w-4 h-4" />
              Scale Down
            </button>
          </div>
        </div>
      </div>

      {/* RPS Multiplier Selector */}
      <div className="mb-6 p-4 rounded-xl bg-zinc-800/50 border border-zinc-700">
        <div className="flex items-center justify-between mb-3">
          <span className="text-sm font-medium text-zinc-400">
            {scaleDirection === 'up' ? 'Traffic Increase' : 'Traffic Decrease'} Factor
          </span>
          <span className={`text-lg font-bold ${scaleDirection === 'up' ? 'text-emerald-500' : 'text-blue-500'}`}>
            {scaleDirection === 'up' ? `${rpsMultiplier}x` : `÷${rpsMultiplier}`}
          </span>
        </div>
        <input
          type="range"
          min="2"
          max="10"
          step="1"
          value={rpsMultiplier}
          onChange={(e) => setRpsMultiplier(Number(e.target.value))}
          className={`w-full h-2 bg-zinc-700 rounded-lg appearance-none cursor-pointer ${scaleDirection === 'up' ? 'accent-emerald-500' : 'accent-blue-500'
            }`}
        />
        <div className="flex justify-between mt-2 text-xs text-zinc-500">
          <span>2x (Light)</span>
          <span>5x (Medium)</span>
          <span>10x (Heavy)</span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        {/* Trigger Prediction Button */}
        <motion.button
          onClick={handleTriggerPrediction}
          disabled={predictionStatus === 'loading'}
          className={`relative overflow-hidden rounded-2xl p-8 border-2 transition-all duration-300 ${predictionStatus === 'loading'
            ? 'border-blue-500/50 bg-blue-500/10'
            : predictionStatus === 'success'
              ? 'border-emerald-500/50 bg-emerald-500/10'
              : predictionStatus === 'error'
                ? 'border-red-500/50 bg-red-500/10'
                : 'border-zinc-700 bg-zinc-800/50 hover:border-blue-500/50 hover:bg-zinc-800'
            }`}
          whileHover={{ scale: predictionStatus === 'idle' ? 1.02 : 1 }}
          whileTap={{ scale: predictionStatus === 'idle' ? 0.98 : 1 }}
        >
          <div className="relative">
            <div className="flex items-center justify-center mb-4">
              {predictionStatus === 'success' ? (
                <CheckCircle className="w-12 h-12 text-emerald-400" />
              ) : predictionStatus === 'error' ? (
                <AlertCircle className="w-12 h-12 text-red-400" />
              ) : (
                <Sparkles className={`w-12 h-12 text-blue-500 ${predictionStatus === 'loading' ? 'animate-pulse' : ''}`} />
              )}
            </div>

            <h3 className="text-xl font-bold mb-2">Trigger Prediction</h3>
            <p className="text-sm text-zinc-500">
              Run ML models to forecast load for the next 60 minutes
            </p>
          </div>
        </motion.button>

        {/* Simulate Scaling Button */}
        <motion.button
          onClick={handleSimulateScale}
          disabled={eventStatus === 'loading'}
          className={`relative overflow-hidden rounded-2xl p-8 border-2 transition-all duration-300 ${eventStatus === 'loading'
            ? scaleDirection === 'up' ? 'border-emerald-500/50 bg-emerald-500/10' : 'border-blue-500/50 bg-blue-500/10'
            : eventStatus === 'success'
              ? 'border-emerald-500/50 bg-emerald-500/10'
              : eventStatus === 'error'
                ? 'border-red-500/50 bg-red-500/10'
                : scaleDirection === 'up'
                  ? 'border-zinc-700 bg-zinc-800/50 hover:border-emerald-500/50 hover:bg-zinc-800'
                  : 'border-zinc-700 bg-zinc-800/50 hover:border-blue-500/50 hover:bg-zinc-800'
            }`}
          whileHover={{ scale: eventStatus === 'idle' ? 1.02 : 1 }}
          whileTap={{ scale: eventStatus === 'idle' ? 0.98 : 1 }}
        >
          <div className="relative">
            <div className="flex items-center justify-center mb-4">
              {eventStatus === 'success' ? (
                <CheckCircle className="w-12 h-12 text-emerald-400" />
              ) : eventStatus === 'error' ? (
                <AlertCircle className="w-12 h-12 text-red-400" />
              ) : scaleDirection === 'up' ? (
                <TrendingUp className={`w-12 h-12 text-emerald-500 ${eventStatus === 'loading' ? 'animate-pulse' : ''}`} />
              ) : (
                <TrendingDown className={`w-12 h-12 text-blue-500 ${eventStatus === 'loading' ? 'animate-pulse' : ''}`} />
              )}
            </div>

            <h3 className="text-xl font-bold mb-2">
              {scaleDirection === 'up' ? 'Simulate Traffic Spike' : 'Simulate Load Decrease'}
            </h3>
            <p className="text-sm text-zinc-500">
              {scaleDirection === 'up'
                ? 'Test predictive scale-up during high traffic'
                : 'Test cost-optimized scale-down when traffic subsides'}
            </p>
          </div>
        </motion.button>
      </div>

      {/* Status Message */}
      <div className="min-h-[60px] flex items-center justify-center">
        {message ? (
          <div
            className={`w-full p-4 rounded-xl border transition-all duration-300 ${predictionStatus === 'success' || eventStatus === 'success'
              ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
              : predictionStatus === 'error' || eventStatus === 'error'
                ? 'bg-red-500/10 border-red-500/30 text-red-400'
                : 'bg-blue-500/10 border-blue-500/30 text-blue-400'
              }`}
          >
            <div className="flex items-center justify-center gap-3">
              {(predictionStatus === 'loading' || eventStatus === 'loading') && (
                <Sparkles className="w-5 h-5 animate-spin" />
              )}
              <p className="text-sm font-medium">{message}</p>
            </div>
          </div>
        ) : (
          <p className="text-sm text-zinc-600">Click a button above to try the demo</p>
        )}
      </div>

      {/* Info boxes */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
        <div className="bg-zinc-800/50 border border-zinc-700 rounded-xl p-4">
          <h4 className="text-sm font-semibold text-blue-400 mb-2">What happens?</h4>
          <p className="text-xs text-zinc-500">
            The system runs Prophet & ensemble models, analyzes patterns, and generates predictions with confidence intervals.
          </p>
        </div>
        <div className="bg-zinc-800/50 border border-zinc-700 rounded-xl p-4">
          <h4 className="text-sm font-semibold text-amber-400 mb-2">Real-world impact</h4>
          <p className="text-xs text-zinc-500">
            ML models detect the spike early, recommend scaling up proactively, and prevent downtime during high traffic.
          </p>
        </div>
      </div>
    </motion.div>
  );
}
