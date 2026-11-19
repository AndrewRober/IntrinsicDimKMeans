namespace KMeansResearchTools;

/// <summary>
/// Contains utility functions for calculations used in K-Means++ and Spectral Intrinsic-Dimension K-Means.
/// </summary>
public static class Utilities
{
    public static PointF TranslatePoint(PointF originalPoint, int width, int height)
    {
        int padding = 20;
        int axisLength = Math.Max(Math.Min(width, height) - 2 * padding, 1);

        float x = (originalPoint.X - padding) * 100f / axisLength;
        float y = 100f - (originalPoint.Y - padding) * 100f / axisLength;

        return new PointF((float)Math.Round(x, 2), (float)Math.Round(y, 2));
    }

    public static PointF ReverseTranslatePoint(PointF translatedPoint, int width, int height)
    {
        int padding = 20;
        int axisLength = Math.Max(Math.Min(width, height) - 2 * padding, 1);

        float x = (translatedPoint.X * axisLength / 100f) + padding;
        float y = padding + (100f - translatedPoint.Y) * axisLength / 100f;

        return new PointF(x, y);
    }

    public static void AssociateToClosestCentroid(List<PointData> points, List<CentroidData> centroids)
    {
        if (centroids.Count == 0)
            return;

        foreach (var point in points)
        {
            CentroidData? closestCentroid = null;
            double minDistance = double.MaxValue;

            foreach (var centroid in centroids)
            {
                double distance = CalculateEuclideanDistance(point.Point, centroid.Point);
                if (distance < minDistance)
                {
                    minDistance = distance;
                    closestCentroid = centroid;
                }
            }

            if (closestCentroid != null)
            {
                point.Centroid = closestCentroid;
                point.Color = closestCentroid.Color;
            }
        }
    }

    public static double CalculateEuclideanDistance(PointF point1, PointF point2)
    {
        double dx = point2.X - point1.X;
        double dy = point2.Y - point1.Y;
        return Math.Sqrt(dx * dx + dy * dy);
    }

    public static PointF CalculateCentroid(List<PointF> points)
    {
        if (points.Count == 0)
            return PointF.Empty;

        float x = 0;
        float y = 0;

        foreach (var point in points)
        {
            x += point.X;
            y += point.Y;
        }

        return new PointF(x / points.Count, y / points.Count);
    }

    /// <summary>
    /// Genuine Arthur & Vassilvitskii (2007) K-Means++ seeding:
    /// samples a new centroid from points with probability proportional to D(x)^2
    /// (squared distance to the nearest existing centroid).
    /// </summary>
    public static PointData SampleKMeansPlusPlusPoint(List<CentroidData> centroids, List<PointData> points, Random rand)
    {
        if (points.Count == 0)
            throw new ArgumentException("Points list cannot be empty.", nameof(points));

        if (centroids.Count == 0)
            return points[rand.Next(points.Count)];

        double[] d2 = new double[points.Count];
        double totalWeight = 0.0;

        for (int i = 0; i < points.Count; i++)
        {
            double minDistSq = double.MaxValue;
            foreach (var c in centroids)
            {
                double dx = points[i].Point.X - c.Point.X;
                double dy = points[i].Point.Y - c.Point.Y;
                double distSq = dx * dx + dy * dy;
                if (distSq < minDistSq)
                    minDistSq = distSq;
            }
            d2[i] = minDistSq;
            totalWeight += minDistSq;
        }

        if (totalWeight <= 1e-12)
            return points[rand.Next(points.Count)];

        double r = rand.NextDouble() * totalWeight;
        double cumulative = 0.0;
        for (int i = 0; i < points.Count; i++)
        {
            cumulative += d2[i];
            if (cumulative >= r)
                return points[i];
        }

        return points[^1];
    }

    /// <summary>
    /// Deterministic Gonzalez (1985) farthest-first traversal (retained for comparison).
    /// </summary>
    public static PointData? GetFurthestPointFromCentroids(List<CentroidData> centroids, List<PointData> points)
    {
        PointData? furthestPoint = null;
        double maxDistance = -1.0;

        foreach (var point in points)
        {
            double minDistanceToCentroid = double.MaxValue;
            foreach (var centroid in centroids)
            {
                double distance = CalculateEuclideanDistance(point.Point, centroid.Point);
                if (distance < minDistanceToCentroid)
                    minDistanceToCentroid = distance;
            }

            if (minDistanceToCentroid > maxDistance)
            {
                maxDistance = minDistanceToCentroid;
                furthestPoint = point;
            }
        }

        return furthestPoint;
    }

    public static void OptimizeCentroidPosition(List<PointData> points, List<CentroidData> centroids)
    {
        foreach (var centroid in centroids)
        {
            var pointsForCentroid = points.Where(p => p.Centroid == centroid).Select(p => p.Point).ToList();
            if (pointsForCentroid.Count > 0)
            {
                centroid.Point = CalculateCentroid(pointsForCentroid);
            }
        }
    }

    public static double CalculateWCSS(CentroidData centroid, List<PointData> points, int scaleWidth, int scaleHeight)
    {
        double wcss = 0;
        var translatedCentroid = TranslatePoint(centroid.Point, scaleWidth, scaleHeight);

        foreach (var point in points.Where(p => p.Centroid == centroid))
        {
            double distance = CalculateEuclideanDistance(
                TranslatePoint(point.Point, scaleWidth, scaleHeight),
                translatedCentroid);
            wcss += distance * distance;
        }

        return wcss;
    }

    public static double CalculateDistance(PointData point1, PointData point2, int width, int height)
    {
        var p1 = TranslatePoint(point1.Point, width, height);
        var p2 = TranslatePoint(point2.Point, width, height);

        double dx = p1.X - p2.X;
        double dy = p1.Y - p2.Y;

        return Math.Sqrt(dx * dx + dy * dy);
    }
}
