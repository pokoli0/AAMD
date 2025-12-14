using System.Collections.Generic;
using UnityEngine;


[System.Serializable]
public struct OHE_Elements
{
    public int position;
    public int count;

    public OHE_Elements(int p, int c)
    {
        position = p;
        count = c;
    }
}

public class OneHotEncoding
{
    List<OHE_Elements> elements;
    Dictionary<int, int> extraElements;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    public OneHotEncoding(List<OHE_Elements> e)
    {
        elements = e;
        extraElements = new Dictionary<int, int>();
        for (int i = 0; i < elements.Count; i++)
        {
            int pos = elements[i].position;
            int c = elements[i].count;
            extraElements.Add(pos, c);
        }
    }

    /// <summary>
    /// Realiza la trasformaci�n del OHE a los elementos seleccionados.
    /// </summary>
    /// <param name="input"></param>
    /// <returns></returns>
    public float[] Transform(float[] input)
    {
        List<float> output = new List<float>();

        for (int i = 0; i < input.Length; i++)
        {
            // Verificamos si requiere OHE
            if (extraElements.ContainsKey(i))
            {
                int categoryCount = extraElements[i];
                int value = (int)input[i]; // Indice actual

                // Vector one-hot
                for (int j = 0; j < categoryCount; j++)
                {
                    output.Add(j == value ? 1.0f : 0.0f);
                }
            }
            else
            {
                output.Add(input[i]);
            }
        }
        return output.ToArray();
    }
}
