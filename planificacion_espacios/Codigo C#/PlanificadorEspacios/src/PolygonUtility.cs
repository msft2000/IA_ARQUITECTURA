using Autodesk.DesignScript.Geometry;


namespace PlanificadorEspacios
{
    public static class PolygonUtility
    {

        // returns area of a closed polygon, if area is positive, poly points are counter clockwise and vice versa
        public static double AreaDynamoPolygon(Polygon poly)
        {
            if (poly==null) return -1;
            List<Point> polyPoints = poly.Points.ToList();
            double area = 0;
            int j = polyPoints.Count - 1;
            for (int i = 0; i < polyPoints.Count; i++)
            {
                area += (polyPoints[j].X + polyPoints[i].X) * (polyPoints[j].Y - polyPoints[i].Y);
                j = i;
            }
            return Math.Abs(area / 2);
        }

    }

}
