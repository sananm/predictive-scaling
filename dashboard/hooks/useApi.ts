import { useState, useEffect, useCallback } from 'react';
import { apiClient } from '@/lib/api';
import type {
  HealthStatus,
  ScalingStatus,
  CurrentPrediction,
  ScalingDecision,
  BusinessEvent,
  SystemConfig,
} from '@/lib/api';

export const useHealth = (interval: number = 5000) => {
  const [data, setData] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const result = await apiClient.getHealth();
        setData(result);
        setError(null);
      } catch (err) {
        setError('Failed to fetch health status');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const intervalId = setInterval(fetchData, interval);
    return () => clearInterval(intervalId);
  }, [interval]);

  return { data, loading, error };
};

export const useScalingStatus = (interval: number = 3000) => {
  const [data, setData] = useState<ScalingStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const result = await apiClient.getScalingStatus();
        setData(result);
        setError(null);
      } catch (err) {
        setError('Failed to fetch scaling status');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const intervalId = setInterval(fetchData, interval);
    return () => clearInterval(intervalId);
  }, [interval]);

  return { data, loading, error };
};

export const usePredictions = (interval: number = 10000) => {
  const [data, setData] = useState<CurrentPrediction | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [recentTriggerResults, setRecentTriggerResults] = useState<any[]>([]);

  const refresh = useCallback(async () => {
    try {
      setLoading(true);
      const result = await apiClient.getCurrentPredictions();
      setData(result);
      setError(null);
    } catch (err) {
      setError('Failed to fetch predictions');
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const intervalId = setInterval(refresh, interval);
    return () => clearInterval(intervalId);
  }, [interval, refresh]);

  const triggerPrediction = useCallback(async () => {
    try {
      console.log('Hook: triggerPrediction called');
      const result = await apiClient.triggerPrediction();
      console.log('Hook: triggerPrediction result', result);

      // Store the predictions from the trigger response
      if (result.predictions && result.predictions.length > 0) {
        setRecentTriggerResults(result.predictions);
        console.log('Hook: stored trigger results', result.predictions.length);
      }

      // Refresh after a delay to get database-stored predictions
      setTimeout(refresh, 2000);
      return result;
    } catch (err) {
      console.error('Hook: triggerPrediction error', err);
      setError('Failed to trigger prediction');
      throw err;
    }
  }, [refresh]);

  return { data, loading, error, refresh, triggerPrediction, recentTriggerResults };
};

export const useScalingDecisions = (limit: number = 10, interval: number = 5000) => {
  const [data, setData] = useState<ScalingDecision[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const result = await apiClient.getScalingDecisions(limit);
        setData(result);
        setError(null);
      } catch (err) {
        setError('Failed to fetch scaling decisions');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const intervalId = setInterval(fetchData, interval);
    return () => clearInterval(intervalId);
  }, [limit, interval]);

  return { data, loading, error };
};

export const useEvents = () => {
  const [data, setData] = useState<BusinessEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setLoading(true);
      const result = await apiClient.getEvents();
      setData(result);
      setError(null);
    } catch (err) {
      setError('Failed to fetch events');
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const createEvent = useCallback(
    async (event: {
      name: string;
      event_type: string;
      start_time: string;
      end_time: string;
      expected_impact_multiplier: number;
    }) => {
      try {
        await apiClient.createEvent(event);
        await refresh();
      } catch (err) {
        setError('Failed to create event');
        console.error(err);
        throw err;
      }
    },
    [refresh]
  );

  return { data, loading, error, refresh, createEvent };
};

export const useConfig = () => {
  const [data, setData] = useState<SystemConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const result = await apiClient.getConfig();
        setData(result);
        setError(null);
      } catch (err) {
        setError('Failed to fetch config');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  return { data, loading, error };
};
