
namespace PlanificadorEspacios
{
    internal class BasicUtility
    {

        //random double numbers between two decimals
        public static double RandomBetweenNumbers(Random rn, double max, double min)
        {
            double num = rn.NextDouble() * (max - min) + min;
            return num;
        }

    }



}
