
using Autodesk.DesignScript.Geometry;


namespace PlanificadorEspacios
{
    /// <summary>
    /// A static class to read contextual data and build the data stack about site outline and program document.
    /// </summary>
    public static class ReadData
    {

        /// <summary>
        /// Generar la estructura de datos para los elementos del espacio
        /// </summary>
        /// <param name="path">Dirección del archivo de planificación.</param>
        /// <returns name="DeptData">Lista de los elementos que representan a los departamentos y sus espacios en base al documento de planificación</returns>
        public static List<DeptData> MakeDataStructure(
            string path = ""
        )
        {
            double dim = 5;//dimension en X e Y por defecto para los programas
            StreamReader reader;
            List<string> deptNameList = new List<string>();
            List<List<string>> progAdjList = new List<List<string>>();
            List<string> progTypeList = new List<string>();
            List<ProgramData> spaces = new List<ProgramData>();

            if (path == "") return null;//validacion de path
            else reader = new StreamReader(File.OpenRead(@path));//abrir el archivo de planificacion

            string docInfo = reader.ReadToEnd();
            string[] csvText = docInfo.Split('\n');
            csvText = csvText.Skip(1).ToArray();//saltar la cabecera
            foreach (string s in csvText)
            {
                if (s.Length == 0) continue;//linea vacia
                var values = s.Split(',');//Divide cada campo del csv
                string nombre_espacio, nombre_departamento, tipo_espacio, adyacencia;
                double area, dimX, dimY;
                int cantidad, preferencia, id;

                id = Convert.ToInt32(values[0]);
                nombre_espacio = values[1];//nombre de espacio
                nombre_departamento = values[2];
                cantidad = Convert.ToInt32(values[3]);//numero de programas de ese tipo
                area = Convert.ToDouble(values[4].Replace(".",","));//area del programa
                preferencia = Convert.ToInt32(values[5]);//valor de preferencia (1 a 10)
                tipo_espacio = values[6];//P: principal y R: Regular
                adyacencia = values[7];//adyacencia entre espacios
                deptNameList.Add(nombre_departamento);//nombre del departamento
                progTypeList.Add(tipo_espacio);//tipo de espacio
                progAdjList.Add(adyacencia.Split('.').ToList());//lista de adyacencia con otros programas

                if (tipo_espacio.ToUpper() == "P")
                {
                    dimX = Convert.ToDouble(values[8].Replace(".", ","));
                    dimY = Convert.ToDouble(values[9].Replace(".", ","));
                }
                else
                {
                    dimX = dim;
                    dimY = dim;
                }
                //Crear la estructura de datos del programa y enlistarla
                spaces.Add(new ProgramData(id, nombre_espacio, nombre_departamento, cantidad,
                    area, preferencia, progAdjList, dimX, dimY, tipo_espacio));//Se agrega el espacio
            }

            List<string> deptNames = deptNameList.Distinct().ToList();//Nombres de los departamentos

            List<DeptData> departments = new List<DeptData>();//lista de departamentos
            List<double> adjWeightList = getSpaceWeight(progTypeList.Count(), progAdjList);//obtener los pesos de cada espacio

            for (int i = 0; i < deptNames.Count; i++)//recorrer los departamentos
            {
                List<ProgramData> spaceInDept = new List<ProgramData>();//lista de espacios por departamento
                //Asignar el peso de cada espacio y colocar ese espacio en el objeto de departamento correspondiente
                for (int j = 0; j < spaces.Count; j++)
                { //recorrer los espacios
                    if (deptNames[i] == spaces[j].DeptName)
                    {
                        spaces[j].AdjacencyWeight = adjWeightList[j];//agregar el peso de adyacencia a cada espacio
                        spaceInDept.Add(spaces[j]);//agregar el espacio a la lista de espacios de un departmento
                    }
                }
                //generar los espacios en base a la cantidad colocada
                List<ProgramData> departmentSpaces = spacesPerQuantity(spaceInDept);
                //generar el departamento
                DeptData dept = new DeptData(deptNames[i], departmentSpaces, 1, dim, dim);
                departments.Add(dept);//se agrega el departamento con los programas actualizados
            }
            //Ordenar los departamentos en base a la preferencia
            List<string> preferredDept = getFrequentDepts(deptNameList, progAdjList, progTypeList);//se calculan los departamentos mas frecuentes

            //ordenar los departamentos por el area utilizada y la importancia
            departments = SortDeptData(departments, preferredDept);

            //calculo de porcentaje de area por departamento
            double totalDeptArea = 0;
            for (int i = 0; i < departments.Count; i++) totalDeptArea += departments[i].DeptAreaNeeded;//area total a utilizar
            for (int i = 0; i < departments.Count; i++) departments[i].DeptAreaProportionNeeded = departments[i].DeptAreaNeeded / totalDeptArea;
            //Asignar los ids propios a cada espacios
            int count = 0;
            List<DeptData> deptDatas = SortProgramsInDept(departments);
            foreach(DeptData d in deptDatas)//ordenar los programas
            {
                foreach(ProgramData p in d.ProgramsInDept)
                {
                    p.OwnProgID = count;
                    count++;
                }
            }
            return deptDatas;
        }


        /// <summary>
        /// Encontrar y ordenar los departamentos por frecuencia de aparicion
        /// </summary>
        /// <param name="deptNameList">nombre de departamento para cada espacio.</param>
        /// <param name="progAdjList">Lista de adyacencia para cada espacio</param>
        /// <param name="progTypeList">Tipo de espacio para cada espacio.</param>
        /// <returns name="depImpList">Devuelve los departamentos ordenados por frecuencia de aparicion.</returns>



        internal static List<string> getFrequentDepts(List<string> deptNameList, List<List<string>> progAdjList, List<string> progTypeList)
        {
            //Procesa y obtiene la lista de departamentos mas frecuentes

            //Obtener la lista de adyacencia de programas con id de programa [[...],[..],...]
            List<List<string>> deptAdjList = new List<List<string>>(progAdjList);
            //foreach (string s in progAdjList) deptAdjList.Add(s.Split('.').ToList());


            List<List<string>> deptNameAdjacencyList = new List<List<string>>();//lista de nombres de departamentos adyacentes para cada espacio [[..],[..]]

            List<string> keyDeptName = new List<string>();//Nombre del departamento que tenga espacios principales (tipo P)
            List<int> keyIndex = new List<int>();//Indice de departamentos que tengan espacios principales (tipo P)

            for (int i = 0; i < progTypeList.Count; i++)
            {//se recorre la cantidad de espacios cargados
                if (progTypeList[i].ToUpper() == "P")
                {
                    //Almacenar el nombre del departamento principal
                    keyDeptName.Add(deptNameList[i]);
                }
            }

            for (int i = 0; i < deptAdjList.Count; i++)
            {//recorrer cada lista de adyacencia de los espacios

                List<string> deptNameAdjacency = new List<string>(); //lista de nombres de departamentos adyacentes al espacio  

                for (int j = 0; j < deptAdjList[i].Count; j++)
                {//Recorre los elementos de adyacencia de cada espacio
                    string str = deptAdjList[i][j];//valor de id de el programa adyacente
                    if (str.Count() < 1 || str == "" || str == " " || str == "\r") str = Convert.ToInt32(BasicUtility.RandomBetweenNumbers(new Random(j), deptNameList.Count - 1, 0)).ToString();//generar un departamento adyacente aleatorio
                    string depName = deptNameList[Convert.ToInt16(str)];//departamento adyacente
                    deptNameAdjacency.Add(depName);//agregar el nombre del departamento a la lista de adyacencia para cada espacio
                }
                deptNameAdjacencyList.Add(deptNameAdjacency);//se agrega la lista de nombres de departamentos adyacentes de cada espacio
            }


            List<string> deptNames = deptNameList.Distinct().ToList();//nombres unicos de departamentos

            //definir los indices de los departamentos principales
            for (int i = 0; i < deptNames.Count; i++) if (keyDeptName.Contains(deptNames[i])) { keyIndex.Add(i); }


            List<List<string>> NumberOfDeptNames = new List<List<string>>();
            List<List<string>> NumberOfDeptTop = new List<List<string>>();

            for (int i = 0; i < deptNames.Count; i++)//Recorrer cada nombre unico de departamento
            {
                List<string> numDeptnames = new List<string>();
                List<string> numDeptTop = new List<string>();

                for (int j = 0; j < deptNameList.Count; j++)//recorrer cada nombre de departamento de cada espacio
                {
                    if (deptNames[i] == deptNameList[j])
                    {
                        numDeptnames.AddRange(deptNameAdjacencyList[j]); //guarda los nombres de departamentos adyacentes para cada espacio
                        numDeptTop.AddRange(deptAdjList[j]); //guarda los ids de espacios adyacentes para cada espacio
                    }
                }
                NumberOfDeptNames.Add(numDeptnames);//PAra cada departamento guarda los nombres de los departamentos que deben ser adyacentes a ellos
                NumberOfDeptTop.Add(numDeptTop);//PAra cada departamento guarda los ids de espacio de los espacios que deben ser adyacentes a ellos
            }


            for (int i = 0; i < deptNames.Count; i++)//recorrer cada nombre de departamento distinto
            {
                NumberOfDeptNames[i].RemoveAll(x => x == deptNames[i]);//eliminar la autoadyacencia al mismo departamento
                //por ver
                NumberOfDeptNames[i].RemoveAll(x => keyDeptName.Contains(x));//eliminar la adyacencia a departamento Principales
                if (keyIndex.Contains(i)) NumberOfDeptNames[i].Clear();//si el departamento es kpu quitar la adyacencia
            }

            List<string> mostFreq = new List<string>();//departamentos mas frecuentes

            for (int i = 0; i < deptNames.Count; i++)//recorre cada lista de nombres de departamentos adyacentes a otro departamento
            {
                var most = "";

                if (NumberOfDeptNames[i].Count == 0) most = "";//si no tiene departamentos adyacentes

                else//si tiene departamentos adacente
                {
                    most = (from item in NumberOfDeptNames[i]
                            group item by item into g
                            orderby g.Count() descending
                            select g.Key).First(); //obtiene el departamento mas recurrente en la lista de adyacencia de cada departamento
                }
                mostFreq.Add(most);  //para cada tipo de departamento guarda el departamento adyacente con mas frecuencia en los programas           
            }

            var frequency = mostFreq.GroupBy(x => x).OrderByDescending(x => x.Count()).ToList();//ordena los departamentos por frecuencia de aparicion

            List<string> depImpList = new List<string>();

            for (int i = 0; i < frequency.Count(); i++) depImpList.AddRange(frequency[i]);//se agregan los departamentos mas frecuentes a la lista

            depImpList = depImpList.Distinct().ToList();//devuelve los nombres de los departamentos mas importantes sin duplicados

            for (int i = 0; i < depImpList.Count(); i++) depImpList.Remove("");//Elimina los espacios vacios

            return depImpList;
        }



        internal static List<double> getSpaceWeight(int numeroEspacios, List<List<string>> progAdjList)
        {
            List<string> progAdjId = new List<string>();
            for (int i = 0; i < numeroEspacios; i++)//Recorrer la lista de programas y sus listas de adyacencia
            {
                //List<string> adjList = progAdjList[i];//dividir por el punto
                progAdjId.AddRange(progAdjList[i]);//agregar los elementos de la lista de string a la nueva lista
            }

            List<int> numIdList = new List<int>();//lista de ids adyacentes a un programa

            //Reemplazar ids de espacios erroneos
            for (int i = 0; i < progAdjId.Count; i++)//Recorrer cada valor de la lista total de adyacencia
            {
                int value;
                try { value = int.Parse(progAdjId[i]); }//convertir string en numero
                catch { value = (int)BasicUtility.RandomBetweenNumbers(new Random(i), numeroEspacios - 1, 0); }//valor de id de programa adyacente aleatorio si la adyacencia colocada no es valida
                numIdList.Add(value);//agregar el id del programa adyacente
            }


            List<double> adjWeightList = new List<double>();//lista de pesos para los programas adyacentes
            for (int i = 0; i < numeroEspacios; i++)//Recorrer el numero de espacios a colocar
            {
                int count = 0;
                for (int j = 0; j < numIdList.Count; j++)//recorrer la lista de ids de programas adyacentes
                {
                    if (i == numIdList[j])
                    {
                        count += 1;
                    }
                }
                adjWeightList.Add(count);//agrega el conteo de veces que cada espacio consta en la lista de adyacencia
            }
            //Se normalizan los pesos a una escala de 0 a 10
            var ratio = 10.0 / adjWeightList.Max();
            adjWeightList = adjWeightList.Select(i => i * ratio).ToList();

            return adjWeightList;
        }


        /// <summary>
        /// Obtener la silueta del sitio de planificacion utilizando un archivo .sat
        /// Devuelve la lista de curvas que forman la silueta del sitio
        /// </summary>
        /// <returns name="geometryList">Lista de curvas que forman la silueta del sitio de planificacion</returns>
        
        public static Geometry[] GetSiteFromSat(string path = "")
        {

            if (path.IndexOf(".sat") != -1)
            {
                return Geometry.ImportFromSAT(path);
            }
            else return null;
        }

        //generar espacios en base a la cantidad especificada en el documento de planificacion
        internal static List<ProgramData> spacesPerQuantity(List<ProgramData> spaces)
        {
            List<ProgramData> spacesQuantity = spaces.Select(x => new ProgramData(x)).ToList();//Copiar el contenido de la lista de espacios
            for (int i = 0; i < spaces.Count; i++)//recorrer cada espacio
                for (int j = 0; j < spaces[i].Quantity - 1; j++) spacesQuantity.Add(spaces[i]);//agregar la cantidad de espacios especifica
            return spacesQuantity.Select(x => new ProgramData(x)).ToList();//Crear objetos nuevos de espacios
        }

        internal struct sortedSpace
        {//estructura para ordenar los espacios por valor de clave
            public double key { get; set; }
            public ProgramData space { get; set; }
        }
        internal class sortedSpaceComparer : IComparer<sortedSpace>
        {//Comparador de departamentos por Area
            public int Compare(sortedSpace x, sortedSpace y)
            {
                return x.key.CompareTo(y.key);
            }
        }
        //ordenar los espacios dentro de un departamento por valor de preferencia
        internal static List<DeptData> SortProgramsInDept(List<DeptData> deptDataInp)
        {
            if (deptDataInp == null) return null;//validar entrada de datos
            List<DeptData> deptData = new List<DeptData>(deptDataInp);//copiar los datos de departmaneto

            double eps = 1, inc = 0.01;
            for (int i = 0; i < deptData.Count; i++)//recorrer los departamentos
            {
                DeptData deptItem = deptData[i];//departamento
                List<ProgramData> spaceItems = deptItem.ProgramsInDept;//espacios dentro del departamento
                List<sortedSpace> sortedSpaces = new List<sortedSpace>();

                for (int j = 0; j < spaceItems.Count; j++)
                {//se recorre cada espacio del departamento
                    double key = spaceItems[j].ProgPreferenceVal + eps + spaceItems[j].AdjacencyWeight;//score en base al valor de preferencia, adyacencia y alpha 
                    sortedSpaces.Add(new sortedSpace { key = key, space = spaceItems[j] });
                    // try { sortedPrograms.Add(key, spaceItems[j]); }//se guarda el epacio con su valor
                    //catch { Random rand = new Random(j);  key += rand.NextDouble(); }//se 
                    spaceItems[j].ProgramCombinedAdjWeight = key;
                    eps += inc;
                }
                sortedSpaces.Sort(new sortedSpaceComparer());//se ordenan los departamentos en base a su metrica de peso combinada con el area
                List<ProgramData> sortedSpacesData = new List<ProgramData>();
                foreach (sortedSpace p in sortedSpaces) sortedSpacesData.Add(p.space);//se guarda cada departamento en orden ascendente
                sortedSpacesData.Reverse();//se cambia el orden de la lista
                eps = 0;
                deptItem.ProgramsInDept = sortedSpacesData;
            }
            return deptData;
        }



        internal struct sortedD
        {//estructura para ordenar los departamentos por area
            public double area { get; set; }
            public DeptData department { get; set; }
        }
        internal class sortedDComparer : IComparer<sortedD>
        {//Comparador de departamentos por Area
            public int Compare(sortedD x, sortedD y)
            {
                return x.area.CompareTo(y.area);
            }
        }
        //Ordenar departamentos en base al area y el peso de sus espacios 
        internal static List<DeptData> SortDeptData(List<DeptData> deptDataInp, List<string> preferredDept)
        {


            List<DeptData> deptData = new List<DeptData>(deptDataInp);
            List<double> areaList = new List<double>();
            List<double> weightList = new List<double>();//peso de cada departamento en base a la adyacencia
            List<string> deptFound = new List<string>();

            List<sortedD> sortedDepartments = new List<sortedD>();//departamentos ordenados

            for (int i = 0; i < preferredDept.Count; i++) weightList.Add(10000000 - (i + 1) * 1000);//recorrer los nombres de los departamentos mas frecuentes



            for (int i = 0; i < deptData.Count; i++)
            {//se recorre cada departamento
                bool match = false;
                for (int j = 0; j < preferredDept.Count; j++)
                {
                    if (preferredDept[j] == deptData[i].DepartmentName)
                    {//se encuentra el departamento
                        areaList.Add(weightList[j]);//se agrega el peso de cada departamento
                        match = true;//si se encuentra el departamento
                        deptFound.Add(preferredDept[j]);//se agrega el departamento
                        deptData[i].DeptAdjacencyWeight = weightList[j];//se agrega el valor de adyacencia en el departamento
                        break;
                    }
                }
                if (!match) { areaList.Add(0); deptFound.Add(""); deptData[i].DeptAdjacencyWeight = areaList[i]; }//Si no se encuentra el departamento
            }

            for (int i = 0; i < deptData.Count; i++)
            {//se recorre cada departamento
                double surpluss = 0;
                double eps = i * BasicUtility.RandomBetweenNumbers(new Random(i), 50, 10);

                if (deptData[i].DepartmentType.ToUpper() == "P")
                    surpluss = 1000000000 + eps + areaList[i];//se le da mas prioridad a los departamentos principales
                else
                    surpluss = areaList[i];//se agrega el peso especificado para cada espacio


                double area = 0.25 * deptData[i].DeptAreaNeeded + surpluss;//se genera un peso tomando en cuenta el area del departamento
                deptData[i].DeptAdjacencyWeight = area;//se agrega el peso al departamento
                sortedDepartments.Add(new sortedD { area = area, department = deptData[i] });//se agrega el departamento a la estructura de comparacion 
            }
            sortedDepartments.Sort(new sortedDComparer());//se ordenan los departamentos en base a su metrica de peso combinada con el area
            List<DeptData> sortedDepartmentData = new List<DeptData>();
            foreach (sortedD p in sortedDepartments) sortedDepartmentData.Add(p.department);//se guarda cada departamento en orden ascendente
            sortedDepartmentData.Reverse();//se cambia el orden de la lista
            return sortedDepartmentData;//se devuelve la lista ordenada
        }

    }
}
