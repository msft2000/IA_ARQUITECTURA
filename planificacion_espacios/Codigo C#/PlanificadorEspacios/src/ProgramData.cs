
using Autodesk.DesignScript.Geometry;
using DSCore;

namespace PlanificadorEspacios
{
    /// <summary>
    /// Program Data object to store information related to program elements from the input program document.
    /// </summary>
    public class ProgramData
    {

        private int _progrID;
        private string _progName;
        private string _progDept;
        private int _progQuantity;
        private double _progUnitArea;
        private int _progPrefValue;
        private List<List<string>> _progAdjList;
        private double _width;//ancho
        private double _height;//alto
        private string _progType;
        private double _areaGiven;
        private PolyCurve _polyProgs;
        private double _adjacencyWeight;
        private double _combinedProgramWeight;
        private int _progrOwnID;
        private Color _color;

        internal ProgramData(int programID, string programName, string programDept,
            int programQuant, double programUnitArea, int programPrefValue, List<List<string>> programAdjList, double width, double height, string progType)
        {
            _progrID = programID;
            _progrOwnID = 0;
            _progDept = programDept;
            _progName = programID + "-" + programName;
            _progQuantity = programQuant;
            _progUnitArea = programUnitArea;
            _progPrefValue = programPrefValue;
            _progAdjList = programAdjList;
            _width = width;
            _height = height;
            _progType = progType;

            _combinedProgramWeight = _progPrefValue;
            _adjacencyWeight = 0;
            _areaGiven = 0;
            _polyProgs = null;
        }

        internal ProgramData(ProgramData other)
        {
            _progrID = other.ProgID;
            _progName = other.ProgramName;
            _progDept = other.DeptName;
            _progQuantity = other.Quantity;
            _progUnitArea = other.UnitArea;
            _progPrefValue = other.ProgPreferenceVal;
            _progAdjList = other.ProgAdjList;
            _width = other._width;
            _height = other._height;
            _progType = other.ProgramType;
            _areaGiven = other.ProgAreaProvided;
            _adjacencyWeight = other.AdjacencyWeight;
            _combinedProgramWeight = other.ProgramCombinedAdjWeight;

            if (other.PolyAssignedToProg != null) _polyProgs = other.PolyAssignedToProg;
            else _polyProgs = null;

        }

        /// <summary>
        /// Name of the program.
        /// </summary>
        public string ProgramName
        {
            get { return _progName; }
            set { _progName = value; }
        }

        /// <summary>
        /// Tipo de Espacio Principal (P) o Regular
        /// </summary>
        public string ProgramType
        {
            get { return _progType; }
        }

        /// <summary>
        /// Ancho del Espacio
        /// </summary>
        public double ProgramWidth
        {
            get { return _width; }
            set { _width = value; }
        }

        /// <summary>
        /// Largo del Espacio
        /// </summary>
        public double ProgramHeight
        {
            get { return _height; }
            set { _height = value; }
        }

        /// <summary>
        /// Polygon assigned to each program.
        /// </summary>
        public PolyCurve PolyAssignedToProg
        {
            get { return _polyProgs; }
            set { _polyProgs = value; }
        }

        internal Color ProgColor
        {
            get { return _color; }
            set { _color = value; }
        }

        /// <summary>
        /// Computed combined program weight.
        /// </summary>
        public double ProgramCombinedAdjWeight
        {
            get { return _combinedProgramWeight; }
            set { _combinedProgramWeight = value; }
        }

        /// <summary>
        /// Computed Adjacency weight value of the program.
        /// </summary>
        public double AdjacencyWeight
        {
            get { return _adjacencyWeight; }
            set { _adjacencyWeight = value; }
        }

        /// <summary>
        /// Program adjacency list.
        /// </summary>
        public List<List<string>> ProgAdjList
        {
            get { return _progAdjList; }
        }

        /// <summary>
        /// Program preference value.
        /// </summary>
        public int ProgPreferenceVal
        {
            get { return _progPrefValue; }
        }

        /// <summary>
        /// Name of the Deparment to which the program is assigned to.
        /// </summary>
        public string DeptName
        {
            get { return _progDept; }
        }

        /// <summary>
        /// Area of one unit of program.
        /// </summary>
        public double UnitArea
        {
            get { return _progUnitArea; }
            set { _progUnitArea = value; }
        }

        /// <summary>
        /// Quantity of each program.
        /// </summary>
        public int Quantity
        {
            get { return _progQuantity; }
        }

        /// <summary>
        /// Area of one unit of the program.
        /// </summary>
        public double ProgAreaNeeded
        {
            get { return _progUnitArea; }
        }


        /// <summary>
        /// Area assigned to the program.
        /// </summary>
        public double ProgAreaProvided
        {//area que se dio al espacio
            get { return _areaGiven; }
            set
            {
                _areaGiven = value;
            }
        }


        internal int ProgID
        {
            get { return _progrID; }
        }
        internal int OwnProgID
        {
            get { return _progrOwnID; }
            set { _progrOwnID = value; }
        }


    }
}
