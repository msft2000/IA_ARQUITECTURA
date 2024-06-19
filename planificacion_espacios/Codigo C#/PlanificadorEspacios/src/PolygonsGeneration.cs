
using Autodesk.DesignScript.Geometry;
using PlanificadorEspacios.src;
using Modifiers;
using DSCore;

namespace PlanificadorEspacios
{
    public static class PolygonsGeneration
    {
        
        public static DeptData[] MakeSpacePolygons(DeptData[] departments, int randomSeed, double minWidth, double minCirculationWidth)
        {
            List<DeptData> departmentsPolys = new List<DeptData>(departments);
            foreach (DeptData d in departmentsPolys)
            {
                d.MakeDepartmentPolygons(randomSeed, minWidth);
                d.DeptMinCirculationWidth = minCirculationWidth/2;
                d.DoCirculationOffset(minCirculationWidth / 2);
            }
            return AssignColors(departmentsPolys.ToArray());
        }
        
        internal static DeptData[] AssignColors(DeptData[] departments)
        {
            List<DeptData> deps = new List<DeptData>(departments);
            int countD = 0;
            int countS = 0;
            foreach(DeptData d in deps){
                d.DeptColor = Colors.getColores[countD];
                countD++;
                foreach(ProgramData s in d.ProgramsInDept)
                {
                    s.ProgColor = Colors.getColores[countS];
                    countS++;
                }
            }
            return deps.ToArray();
        }

        public static DeptData[] UndoPolygonsCirculationOffset(DeptData[] departments)
        {
            List<DeptData> departmentsPolys = new List<DeptData>(departments);
            foreach (DeptData d in departmentsPolys)
            {
                d.UndoCirculationOffset();
            }
            return departmentsPolys.ToArray();
        }

        public static Dictionary<string, object> GetSpacesIdPolygons(DeptData[] departments)
        {
            List<int> ids= new List<int>();
            List<double> adjW = new List<double>();
            List<PolyCurve> polygons=new List<PolyCurve>();
            foreach (DeptData d in departments)
            {
                foreach(ProgramData s in d.ProgramsInDept)
                {
                    ids.Add(s.OwnProgID);
                    adjW.Add(s.ProgramCombinedAdjWeight);
                    polygons.Add(s.PolyAssignedToProg);
                }
            }
            return new Dictionary<string, object>
            {
                { "SpacesIDs", ids },
                { "SpacesAdjWeigth", adjW },
                { "SpacePolygon",polygons}
            };
        }
    
        public static List<DeptData> AssingPolygons(DeptData[] departments,List<int> ids, List<PolyCurve> polygons)
        {
            List<DeptData> deps = new List<DeptData>(departments);
            foreach (DeptData d in deps)
            {
                foreach(ProgramData p in d.ProgramsInDept)
                {
                    int idx=ids.IndexOf(p.OwnProgID);
                    if(idx>=0)
                    {
                        p.PolyAssignedToProg = polygons[idx];
                        p.ProgAreaProvided = PolygonUtility.AreaDynamoPolygon(Polygon.ByPoints(p.PolyAssignedToProg.Points));
                    }
                    else{
                        p.PolyAssignedToProg = null;
                    }
                }
            }
            return deps;
        }

        public static Dictionary<string, object> showAssignedCurves(DeptData[] departments, Boolean onlyDepartaments = false)
        {
            List<Modifiers.GeometryColor> polygons = new List<Modifiers.GeometryColor>();
            List<PolyCurve> borders = new List<PolyCurve>();
            List<string> names = new List<string>();
            foreach (DeptData d in departments)
            {
                foreach (ProgramData p in d.ProgramsInDept)
                {
                    if (p.PolyAssignedToProg != null)
                    {
                        Color color = p.ProgColor;
                        if (onlyDepartaments) color = d.DeptColor;//Para solo mostar el color del departamento
                        //De polygono a superficie
                        Surface surf = Surface.ByPatch(p.PolyAssignedToProg);
                        Modifiers.GeometryColor dis = Modifiers.GeometryColor.ByGeometryColor(surf, color);
                        polygons.Add(dis);
                        borders.Add(p.PolyAssignedToProg);
                        names.Add(p.ProgramName);
                    }
                }
            }
            return new Dictionary<string, object>
            {

                { "ColorPolygons", polygons },
                { "Borders", borders },
                { "SpaceNames",names}
            };
        }
    }
}