using Autodesk.DesignScript.Geometry;

namespace PlanificadorEspacios.src
{
    public static class Analysis
    {
        public static Dictionary<string,Object> UtilizedSpace(double usableArea, DeptData[] departments)
        {
            double area_used = 0;
            int used_spaces = 0;
            int total_spaces = 0;
            double adjacency_score = 0;
            foreach(DeptData i in new List<DeptData>(departments))
            {
                foreach(ProgramData p in i.ProgramsInDept)
                {
                    total_spaces++;
                    if (p.PolyAssignedToProg != null)
                    {
                        used_spaces++;
                        area_used += p.ProgAreaProvided;
                        adjacency_score += p.ProgramCombinedAdjWeight;
                    }
                }
            }
            return new Dictionary<string, object>
            {
                {"Percentage_AreaUsed", area_used/usableArea},
                {"Percentage_SpacesPlaced", (double) used_spaces/total_spaces},
                {"CombinedAdjWeight",adjacency_score}
            };
        }
    
        public static Dictionary<string, Object> GeneratePoints(PolyCurve[] obstacles, PolyCurve[] spaces,double offset)
        {
            List<PolyCurve> obs_offset=new List<PolyCurve>();
            List<PolyCurve> spa_offset = new List<PolyCurve>();
            List<Curve> curves = new List<Curve>();
            List<Point> puntos_obstaculos= new List<Point>();  
            List<List<Point>> puntos_espacios_group=new List<List<Point>>();
            List<Point> puntos_espacios = new List<Point>();
            foreach (PolyCurve obstacle in obstacles)
            {          
                puntos_obstaculos.Add(Polygon.ByPoints(obstacle.Points).Center());
            }
            foreach (PolyCurve space in spaces)
            {
                curves.AddRange(space.Curves());
                List <Point> puntos = new List<Point>();
                foreach (Curve curve in ((PolyCurve)space.OffsetMany(offset, null)[0]).Curves())
                {
                    puntos_espacios.Add(curve.PointAtParameter(0.5));
                }
            }
            for (int i = 0; i < puntos_obstaculos.Count; i++)
            {
                puntos_espacios_group.Add(puntos_espacios);
            }

            return new Dictionary<string, object> { { "Curves", curves }, { "Points_Obs", puntos_obstaculos }, { "Points_Spa", puntos_espacios_group } };
        }
    }
}
