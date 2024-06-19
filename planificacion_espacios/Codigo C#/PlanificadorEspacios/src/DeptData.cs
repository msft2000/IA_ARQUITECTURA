
using Autodesk.DesignScript.Geometry;
using DSCore;


namespace PlanificadorEspacios
{
    public class DeptData
    {
        private string _deptName;
        private List<ProgramData> _progDataList;
        private double _deptAreaNeeded;
        private double _areaGivenDept;
        private double _gridX;
        private double _gridY;
        private string _deptType;
        private List<Polygon> _polyDepts;
        private double _deptAreaProportion;
        private double _deptAreaProportionAchieved;
        private double _minCirculationWidth;
        private double _deptAdjacencyWeight;
        private Color _color;

        internal DeptData(string deptName, List<ProgramData> programDataList, double circulationFactor, double dimX, double dimY)
        {
            _deptName = deptName;
            _progDataList = programDataList;
            _minCirculationWidth = circulationFactor;
            _deptAreaNeeded = AreaEachDept();
            _gridX = dimX;
            _gridY = dimY;
            _deptType = CalcDepartmentType();
            _polyDepts = null;
            _deptAreaProportion = 0;
            _deptAreaProportionAchieved = 0;
            _deptAdjacencyWeight = 0;

        }

        internal DeptData(DeptData other)
        {
            _deptName = other.DepartmentName;
            _progDataList = other.ProgramsInDept;
            _minCirculationWidth = other.DeptMinCirculationWidth;
            _deptAreaNeeded = other.AreaEachDept();
            _deptAreaNeeded = other.DeptAreaNeeded;
            _gridX = other._gridX;
            _gridY = other._gridY;
            _deptType = other.DepartmentType;
            _deptAreaProportion = other.DeptAreaProportionNeeded;
            _deptAreaProportionAchieved = other.DeptAreaProportionAchieved;

            _areaGivenDept = other.DeptAreaProvided;
            _deptAdjacencyWeight = other.DeptAdjacencyWeight;

            if (other.PolyAssignedToDept != null && other.PolyAssignedToDept.Count > 0) _polyDepts = other.PolyAssignedToDept;
            else _polyDepts = null;
        }


        /// <summary>
        /// Required area proportion for each department on site.
        /// </summary>
        public double DeptAreaProportionNeeded
        {
            get { return _deptAreaProportion; }
            set { _deptAreaProportion = value; }
        }

        /// <summary>
        /// Department Adjacency Weight.
        /// </summary>
        public double DeptAdjacencyWeight
        {
            get { return _deptAdjacencyWeight; }
            set { _deptAdjacencyWeight = value; }
        }

        internal Color DeptColor
        {
            get { return _color; }
            set { _color = value; }
        }


        /// <summary>
        /// Returns the area proportion achieved for each department after space plan layout is generated.
        /// </summary>
        public double DeptAreaProportionAchieved
        {
            get { return _deptAreaProportionAchieved; }
            set { _deptAreaProportionAchieved = value; }
        }

        /// <summary>
        /// Type of Department (either KPU or Regular ).
        /// </summary>
        public string DepartmentType
        {
            get { return _deptType; }
        }


        /// <summary>
        /// Polygon2d assigned to each department.
        /// </summary>     
        public List<Polygon> PolyAssignedToDept
        {
            get { return _polyDepts; }
            set { _polyDepts = value; }
        }


        /// <summary>
        /// Area provided to each department.
        /// </summary>
        public double DeptAreaProvided
        {
            get { return _areaGivenDept; }
            set
            {
                _areaGivenDept = value;
            }
        }

        /// <summary>
        /// Name of the Department.
        /// </summary>
        public string DepartmentName
        {
            get { return _deptName; }
        }

        /// <summary>
        /// Area needed for each department.
        /// </summary>
        public double DeptAreaNeeded
        {
            get { return _deptAreaNeeded; }
        }

        /// <summary>
        /// List of programs inside each department.
        /// </summary>
        public List<ProgramData> ProgramsInDept
        {
            get { return _progDataList; }
            set { _progDataList = value; }
        }

        internal void DoCirculationOffset(double offset)
        {
            foreach (ProgramData p in _progDataList)
            {
                PolyCurve newPoly = (PolyCurve) p.PolyAssignedToProg.OffsetMany(offset,null)[0];
                p.PolyAssignedToProg = newPoly;
            }
        }
        internal void UndoCirculationOffset()
        {
            foreach (ProgramData p in _progDataList)
            {
                if (p.PolyAssignedToProg != null)
                {
                    PolyCurve newPoly = (PolyCurve)p.PolyAssignedToProg.OffsetMany(-DeptMinCirculationWidth, null)[0];
                    p.PolyAssignedToProg = newPoly;
                }
            }
        }

        internal void MakeDepartmentPolygons(int seed, double minWidth)
        {
            foreach (ProgramData p in _progDataList)
            {
                if (p.ProgramType == "P")
                {//El area ya esta definida

                    p.PolyAssignedToProg = PolyCurve.ByPoints(MakePointsByWH(p.ProgramWidth, p.ProgramHeight));
                }
                else
                {
                    (double w, double h) = GenerateWH(p.UnitArea, minWidth, seed);
                    p.PolyAssignedToProg = PolyCurve.ByPoints(MakePointsByWH(w, h));
                    p.ProgramWidth = w;
                    p.ProgramHeight = h;
                }
            }
        }

        internal List<Point> MakePointsByWH(double w, double h)
        {
            List<Point> points = new List<Point>
            {
                Point.ByCoordinates(0, 0),
                Point.ByCoordinates(0, h),
                Point.ByCoordinates(w, h),
                Point.ByCoordinates(w, 0),
                Point.ByCoordinates(0, 0)
            };
            return points;
        }

        internal (double width, double height) GenerateWH(double area, double minWidth, int seed)
        {
            double ASPECT_RATIO = 1.618;
            int count = 0;
            double maxWidth, selectedWidth = 0, newHeight = 0;
            double bestWidth = 0, bestHeight = 0;
            double bestDiff = double.MaxValue;
            Random rnd = new Random(seed);
            Boolean solution = false;
            maxWidth = System.Math.Sqrt(area);

            while (count < 50 && !solution)
            {
                selectedWidth = minWidth + (maxWidth - minWidth) * rnd.NextDouble();
                newHeight = area / selectedWidth;
                double diff = System.Math.Abs(selectedWidth / newHeight - ASPECT_RATIO);
                if (diff < bestDiff)
                {
                    bestWidth = selectedWidth;
                    bestHeight = newHeight;
                    bestDiff = diff;
                }
                if (selectedWidth / newHeight >= ASPECT_RATIO)
                {
                    solution = true;
                }
                else
                {
                    count++;
                }
            }

            return (bestWidth, bestHeight);
        }

        //dept circulation factor
        internal double DeptMinCirculationWidth
        {
            get { return _minCirculationWidth; }
            set { _minCirculationWidth = value; }
        }

        internal double AreaEachDept()
        {
            if (_progDataList == null) return 0;
            double area = 0;
            for (int i = 0; i < _progDataList.Count; i++) area += _progDataList[i].UnitArea;
            return area;
        }

        internal string CalcDepartmentType()
        {
            if (_progDataList == null) return "";
            int count = 0;
            for (int i = 0; i < _progDataList.Count; i++)
                if (_progDataList[i].ProgramType.ToUpper() == "P") count += 1;
            int perc = count / _progDataList.Count;
            if (perc > 0.50) return "P";
            else return "R";
        }

    }
}

