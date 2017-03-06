using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;

class MultiplyMatrices
{
    #region Sequential_Loop
    static void MultiplyMatricesSequential(double[,] matA, double[,] matB,
                                            double[,] result)
    {
        long matACols = matA.GetLength(1);
        long matBCols = matB.GetLength(1);
        long matARows = matA.GetLength(0);

        for (long i = 0; i < matARows; i++)
        {
            for (long j = 0; j < matBCols; j++)
            {
                double temp = 0;
                for (long k = 0; k < matACols; k++)
                {
                    temp += matA[i, k] * matB[k, j];
                }
                result[i, j] += temp;
            }
        }
    }
    #endregion

    #region Parallel_Loop
    static void MultiplyMatricesParallel(double[,] matA, double[,] matB, double[,] result)
    {
        long matACols = matA.GetLength(1);
        long matBCols = matB.GetLength(1);
        long matARows = matA.GetLength(0);

        // A basic matrix multiplication.
        // Parallelize the outer loop to partition the source array by rows.
        Parallel.For(0, matARows, i =>
        {
            for (long j = 0; j < matBCols; j++)
            {
                double temp = 0;
                for (long k = 0; k < matACols; k++)
                {
                    temp += matA[i, k] * matB[k, j];
                }
                result[i, j] = temp;
            }
        }); // Parallel.For
    }
    #endregion

    static Tuple<List<long>, List<long>> Iterate(long colCount, long rowCount, int itertations = 5)
    {
        double[,] m1 = InitializeMatrix(rowCount, colCount);
        double[,] m2 = InitializeMatrix(colCount, rowCount);
        double[,] result = new double[rowCount, rowCount];

        List<long> sequentials = new List<long>();
        List<long> parallels = new List<long>();

        Stopwatch stopwatch = new Stopwatch();

        for (int i = 0; i < itertations; i++)
        {
            stopwatch.Reset();
            // sequential
            stopwatch.Start();
            MultiplyMatricesSequential(m1, m2, result);
            stopwatch.Stop();
            sequentials.Add(stopwatch.ElapsedMilliseconds);

            stopwatch.Reset();
            // parallel
            stopwatch.Start();
            MultiplyMatricesParallel(m1, m2, result);
            stopwatch.Stop();
            parallels.Add(stopwatch.ElapsedMilliseconds);
        }

        return Tuple.Create<List<long>, List<long>>(sequentials, parallels);
    }

    #region Main
    static void Main(string[] args)
    {
        int from = 1;
        int to = 11;
        List<int> powersOf2 = Enumerable.Range(from, to).Select(power => 1 << power).ToList<int>();

        using (System.IO.StreamWriter file = new System.IO.StreamWriter("results.csv", true))
        {
            foreach (var power in powersOf2)
            {
                Tuple<List<long>, List<long>> times = Iterate(power, power, 3);
                foreach (var time in times.Item1)
                {
                    file.WriteLine(String.Format("sequential,{0},{0},{1}", power, time));
                }
                foreach (var time in times.Item2)
                {
                    file.WriteLine(String.Format("parallel,{0},{0},{1}", power, time));
                }
                file.Flush();
            }
        }

        // Keep the console window open in debug mode.
        Console.Error.WriteLine("Press any key to exit.");
        Console.ReadKey();
    }
    #endregion

    #region Helper_Methods
    static double[,] InitializeMatrix(long rows, long cols)
    {
        double[,] matrix = new double[rows, cols];

        Random r = new Random();
        for (long i = 0; i < rows; i++)
        {
            for (long j = 0; j < cols; j++)
            {
                matrix[i, j] = r.Next(100);
            }
        }
        return matrix;
    }
    #endregion
}
