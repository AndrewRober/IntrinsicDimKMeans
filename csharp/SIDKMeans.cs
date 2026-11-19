namespace KMeansResearchTools;

/// <summary>
/// C# Implementation of Spectral Intrinsic-Dimension Minimum Description Length K-Means (SID-KMeans).
/// Computes the Satterthwaite-Welch effective degrees of freedom nu = (Tr Sigma)^2 / Tr(Sigma^2)
/// via Frobenius norm identity Tr(Sigma^2) = ||(1/N) X_c^T X_c||_F^2 in O(N d^2) (or O(N^2 d)),
/// and selects K* in {1..K_max} via exact LogSumExp marginal mixture MDL without any tunable lambda.
/// </summary>
public static class SIDKMeansSelector
{
    public sealed record SelectionResult(int OptimalK, double EffectiveDoF, double[] ScoresByK);

    public static SelectionResult Evaluate2DPoints(List<PointF> points, int kMax = 10, int seed = 42)
    {
        int n = points.Count;
        if (n < 3)
            return new SelectionResult(1, 1.0, new double[] { 0.0 });

        double[,] X = new double[n, 2];
        for (int i = 0; i < n; i++)
        {
            X[i, 0] = points[i].X;
            X[i, 1] = points[i].Y;
        }
        return EvaluateMatrix(X, kMax, seed);
    }

    public static SelectionResult EvaluateMatrix(double[,] X, int kMax = 12, int seed = 42)
    {
        int n = X.GetLength(0);
        int d = X.GetLength(1);
        int kUpper = Math.Min(kMax, n - 1);

        // 1. Mean-center X
        double[] mean = new double[d];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++)
                mean[j] += X[i, j] / n;

        double[,] Xc = new double[n, d];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++)
                Xc[i, j] = X[i, j] - mean[j];

        // 2. Exact Satterthwaite-Welch Effective Degrees of Freedom:
        //    Tr(Sigma) = (1/N) sum_{i,j} Xc_{i,j}^2,  Tr(Sigma^2) = ||(1/N) Xc^T Xc||_F^2
        double trSigma = 0.0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++)
                trSigma += (Xc[i, j] * Xc[i, j]) / n;

        double trSigmaSq = 0.0;
        for (int j1 = 0; j1 < d; j1++)
        {
            for (int j2 = 0; j2 < d; j2++)
            {
                double cov12 = 0.0;
                for (int i = 0; i < n; i++)
                    cov12 += Xc[i, j1] * Xc[i, j2];
                cov12 /= n;
                trSigmaSq += cov12 * cov12;
            }
        }

        if (trSigmaSq <= 1e-24)
            return new SelectionResult(1, 1.0, new double[] { 0.0 });

        double nu = Math.Clamp((trSigma * trSigma) / trSigmaSq, 1.0, d);
        double[] scores = new double[kUpper];
        Random rng = new Random(seed);

        for (int k = 1; k <= kUpper; k++)
        {
            (int[] labels, double[,] centers, double wcss) = RunKMeansPlusPlus(Xc, k, rng);
            wcss = Math.Max(wcss, 1e-18);

            int[] counts = new int[k];
            for (int i = 0; i < n; i++)
                counts[labels[i]]++;

            double sig2Eff = wcss / (n * nu);
            double neg2LogLik = 0.0;

            for (int i = 0; i < n; i++)
            {
                double[] logTerms = new double[k];
                double maxLog = double.NegativeInfinity;

                for (int c = 0; c < k; c++)
                {
                    double piC = Math.Max((double)counts[c] / n, 1e-15);
                    double sqDist = 0.0;
                    for (int j = 0; j < d; j++)
                    {
                        double diff = Xc[i, j] - centers[c, j];
                        sqDist += diff * diff;
                    }
                    double val = Math.Log(piC) - 0.5 * nu * Math.Log(2.0 * Math.PI * sig2Eff) - sqDist / (2.0 * sig2Eff);
                    logTerms[c] = val;
                    if (val > maxLog)
                        maxLog = val;
                }

                double sumExp = 0.0;
                for (int c = 0; c < k; c++)
                    sumExp += Math.Exp(logTerms[c] - maxLog);

                neg2LogLik += -2.0 * (maxLog + Math.Log(sumExp));
            }

            double penalty = (k * (nu + 1.0) - 1.0) * Math.Log(n);
            scores[k - 1] = neg2LogLik + penalty;
        }

        int bestIdx = 0;
        for (int idx = 1; idx < kUpper; idx++)
        {
            if (scores[idx] < scores[bestIdx])
                bestIdx = idx;
        }

        return new SelectionResult(bestIdx + 1, nu, scores);
    }

    private static (int[] labels, double[,] centers, double wcss) RunKMeansPlusPlus(double[,] X, int k, Random rng)
    {
        int n = X.GetLength(0);
        int d = X.GetLength(1);
        int[] labels = new int[n];
        double[,] centers = new double[k, d];

        if (k == 1)
        {
            double wcss1 = 0.0;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < d; j++)
                    wcss1 += X[i, j] * X[i, j];
            return (labels, centers, wcss1);
        }

        // K-Means++ initialization
        int first = rng.Next(n);
        for (int j = 0; j < d; j++)
            centers[0, j] = X[first, j];

        double[] minDistSq = new double[n];
        for (int c = 1; c < k; c++)
        {
            double sumWeights = 0.0;
            for (int i = 0; i < n; i++)
            {
                double best = double.MaxValue;
                for (int prev = 0; prev < c; prev++)
                {
                    double distSq = 0.0;
                    for (int j = 0; j < d; j++)
                    {
                        double diff = X[i, j] - centers[prev, j];
                        distSq += diff * diff;
                    }
                    if (distSq < best)
                        best = distSq;
                }
                minDistSq[i] = best;
                sumWeights += best;
            }

            int chosen = n - 1;
            if (sumWeights > 1e-18)
            {
                double r = rng.NextDouble() * sumWeights;
                double acc = 0.0;
                for (int i = 0; i < n; i++)
                {
                    acc += minDistSq[i];
                    if (acc >= r)
                    {
                        chosen = i;
                        break;
                    }
                }
            }
            for (int j = 0; j < d; j++)
                centers[c, j] = X[chosen, j];
        }

        // Lloyd iterations
        for (int iter = 0; iter < 35; iter++)
        {
            bool changed = false;
            for (int i = 0; i < n; i++)
            {
                int bestC = 0;
                double bestD = double.MaxValue;
                for (int c = 0; c < k; c++)
                {
                    double dSq = 0.0;
                    for (int j = 0; j < d; j++)
                    {
                        double diff = X[i, j] - centers[c, j];
                        dSq += diff * diff;
                    }
                    if (dSq < bestD)
                    {
                        bestD = dSq;
                        bestC = c;
                    }
                }
                if (labels[i] != bestC)
                {
                    labels[i] = bestC;
                    changed = true;
                }
            }

            double[,] sums = new double[k, d];
            int[] cnts = new int[k];
            for (int i = 0; i < n; i++)
            {
                int c = labels[i];
                cnts[c]++;
                for (int j = 0; j < d; j++)
                    sums[c, j] += X[i, j];
            }
            for (int c = 0; c < k; c++)
            {
                if (cnts[c] > 0)
                {
                    for (int j = 0; j < d; j++)
                        centers[c, j] = sums[c, j] / cnts[c];
                }
            }
            if (!changed && iter > 0)
                break;
        }

        double wcss = 0.0;
        for (int i = 0; i < n; i++)
        {
            int c = labels[i];
            for (int j = 0; j < d; j++)
            {
                double diff = X[i, j] - centers[c, j];
                wcss += diff * diff;
            }
        }
        return (labels, centers, wcss);
    }
}
