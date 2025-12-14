using System.Collections.Generic;
using UnityEngine;

public class MLPParameters
{
    List<float[,]> coeficients;
    List<float[]> intercepts;

    public MLPParameters(int numLayers)
    {
        coeficients = new List<float[,]>();
        intercepts = new List<float[]>();
        for (int i = 0; i < numLayers - 1; i++)
        {
            coeficients.Add(null);
        }
        for (int i = 0; i < numLayers - 1; i++)
        {
            intercepts.Add(null);
        }
    }

    public void CreateCoeficient(int i, int rows, int cols)
    {
        coeficients[i] = new float[rows, cols];
    }

    public void SetCoeficiente(int i, int row, int col, float v)
    {
        coeficients[i][row, col] = v;
    }

    public List<float[,]> GetCoeff()
    {
        return coeficients;
    }
    public void CreateIntercept(int i, int row)
    {
        intercepts[i] = new float[row];
    }

    public void SetIntercept(int i, int row, float v)
    {
        intercepts[i][row] = v;
    }
    public List<float[]> GetInter()
    {
        return intercepts;
    }
}

public class MLPModel
{
    MLPParameters mlpParameters;
    public MLPModel(MLPParameters p)
    {
        mlpParameters = p;
    }

    /// <summary>
    /// Parameters required for model input. By default it will be perception, kart position and time, 
    /// but depending on the data cleaning and data acquisition modificiations made by each one, the input will need more parameters.
    /// </summary>
    /// <param name="p">The Agent perception</param>
    /// <returns>The action label</returns>
    public float[] FeedForward(float[] input)
    {
        List<float[,]> weights = mlpParameters.GetCoeff();
        List<float[]> intercepts = mlpParameters.GetInter();

        float[] currentLayerInput = input;

        // Iteramos por cada capa de la red
        for (int i = 0; i < weights.Count; i++)
        {
            // Matriz de pesos
            float[,] w = weights[i];  
            // Vector de bias
            float[] b = intercepts[i];  
            
            int nInputs = w.GetLength(0);
            int nOutputs = w.GetLength(1);

            float[] layerOutput = new float[nOutputs];

            // Input * Pesos + Bias
            for (int j = 0; j < nOutputs; j++)
            {
                // Inicializamos con el bias
                float sum = b[j]; 

                for (int k = 0; k < nInputs; k++)
                {
                    if (k < currentLayerInput.Length) 
                    {
                        sum += currentLayerInput[i] * w[k, j];
                    }
                }

                // Aplicamos función de activación
                layerOutput[j] = sigmoid(sum);
            }

            // La salida de esta capa es la entrada de la siguiente
            currentLayerInput = layerOutput;
        }

        // Al final aplicamos Softmax
        return SoftMax(currentLayerInput);
    }

    /// <summary>
    /// Calculo de la sigmoidal
    /// </summary>
    /// <param name="z"></param>
    /// <returns></returns>
    private float sigmoid(float z)
    {
        return 1.0f / (1.0f + Mathf.Exp(-z));
    }


    /// <summary>
    /// CAlculo de la soft max, se le pasa el vector de la ulrima capa oculta y devuelve el mismo vector, pero procesado
    /// aplicando softmax a cada uno de los elementos
    /// </summary>
    /// <param name="zArr"></param>
    /// <returns></returns>
    public float[] SoftMax(float[] zArr)
    {
        float[] result = new float[zArr.Length];
        float sum = 0f;

        // Calculamos exponenciales y suma total
        for (int i = 0; i < zArr.Length; i++)
        {
            result[i] = Mathf.Exp(zArr[i]);
            sum += result[i];
        }

        // Dividimos cada elemento por la suma
        for (int i = 0; i < zArr.Length; i++)
        {
            if (sum != 0)
                result[i] /= sum;
        }

        return result;
    }

    /// <summary>
    /// Elige el output de mayor nivel
    /// </summary>
    /// <param name="output"></param>
    /// <returns></returns>
    public int Predict(float[] output)
    {
        float max;
        int index = GetIndexMaxValue(output, out max);
        return index;
    }

    /// <summary>
    /// Obtiene el �ndice de mayor valor.
    /// </summary>
    /// <param name="output"></param>
    /// <param name="max"></param>
    /// <returns></returns>
    public int GetIndexMaxValue(float[] output, out float max)
    {
        max = -float.MaxValue;
        int index = 0;

        for (int i = 0; i < output.Length; i++)
        {
            if (!(output[i] > max)) continue;
            
            max = output[i];
            index = i;
        }
        return index;
    }
}
