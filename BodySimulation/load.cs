using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Newtonsoft.Json;

public class DataMats
{
    public float [] X_mu_vec {set;get;}
    public float [] Y_mu_vec {set;get;}
    public float [,] U {set;get;}
    public float [,] V {set;get;}
    public float [] alpha {set;get;}
    public float [] beta {set;get;}
    public float [,] Q {set;get;}
    public float [] norms_mu_vec {set;get;}
    public float [,] U_T {set;get;}
    public float [,] Q_T {set;get;}
}

public class DataMatsEigen
{
    public Matrix X_mu_vec {set;get;}
    public Matrix Y_mu_vec {set;get;}
    public Matrix U {set;get;}
    public Matrix V {set;get;}
    public Matrix alpha {set;get;}
    public Matrix beta {set;get;}
    public Matrix Q {set;get;}
    public Matrix norms_mu_vec {set;get;}
    public Matrix U_T {set;get;}
    public Matrix Q_T {set;get;}    
}

public class ModelMats
{
    public float [,] fc1_weight {set;get;}
    public float [] fc1_bias {set;get;}
    public float [,] fc2_weight {set;get;}
    public float [] fc2_bias {set;get;}
    public float [,] fc3_weight {set;get;}
    public float [] fc3_bias {set;get;}
    public float [,] fc4_weight {set;get;}
    public float [] fc4_bias {set;get;}
    public float [,] fc5_weight {set;get;}
    public float [] fc5_bias {set;get;}
    public float [,] fc6_weight {set;get;}
    public float [] fc6_bias {set;get;}
    public float [,] fc7_weight {set;get;}
    public float [] fc7_bias {set;get;}
    public float [,] fc8_weight {set;get;}
    public float [] fc8_bias {set;get;}
    public float [,] fc9_weight {set;get;}
    public float [] fc9_bias {set;get;}
    public float [,] output_weight {set;get;}
    public float [] output_bias {set;get;}
}

public class ModelMatsEigen
{
    public Matrix fc1_weight {set;get;}
    public Matrix fc1_bias {set;get;}
    public Matrix fc2_weight {set;get;}
    public Matrix fc2_bias {set;get;}
    public Matrix fc3_weight {set;get;}
    public Matrix fc3_bias {set;get;}
    public Matrix fc4_weight {set;get;}
    public Matrix fc4_bias {set;get;}
    public Matrix fc5_weight {set;get;}
    public Matrix fc5_bias {set;get;}
    public Matrix fc6_weight {set;get;}
    public Matrix fc6_bias {set;get;}
    public Matrix fc7_weight {set;get;}
    public Matrix fc7_bias {set;get;}
    public Matrix fc8_weight {set;get;}
    public Matrix fc8_bias {set;get;}
    public Matrix fc9_weight {set;get;}
    public Matrix fc9_bias {set;get;}
    public Matrix output_weight {set;get;}
    public Matrix output_bias {set;get;} 
}

public class TestLoad
{
    public DataMats dataMats;
    public ModelMats modelMats;
    public DataMatsEigen dataMatsEigen = new DataMatsEigen();
    public ModelMatsEigen modelMatsEigen = new ModelMatsEigen();
    public TestLoad()
    {
        // data_json_path = "D:/project/Unity Project/FlagSim/Assets/Datas/data_json.json";
        // model_json_path = "D:/project/Unity Project/FlagSim/Assets/Datas/model_json.json"; 
        TextAsset dataAsset = Resources.Load<TextAsset>("data_json");
        dataMats = JsonConvert.DeserializeObject<DataMats>(dataAsset.text);
        TextAsset modelAsset = Resources.Load<TextAsset>("model_json");
        modelMats = JsonConvert.DeserializeObject<ModelMats>(modelAsset.text);
        Debug.Log(dataMats);
        Debug.Log(modelMats);

        InitEigenMats();
        SetEigenMatsValues();
        // Debug.Log(dataMats.alpha[127]);
        // Debug.Log(dataMats.U[127, 11]);
        // Debug.Log(modelMats.fc9_bias[3]);
    }

    public void InitEigenMats()
    {
        Debug.Log(dataMats.X_mu_vec.GetLength(0));
        dataMatsEigen.X_mu_vec = new Matrix(1, dataMats.X_mu_vec.GetLength(0), "X_mu_vec");
        dataMatsEigen.Y_mu_vec = new Matrix(1, dataMats.Y_mu_vec.GetLength(0), "Y_mu_vec");
        dataMatsEigen.U = new Matrix(dataMats.U.GetLength(0), dataMats.U.GetLength(1), "U");
        dataMatsEigen.V = new Matrix(dataMats.V.GetLength(0), dataMats.V.GetLength(1), "V");
        dataMatsEigen.alpha = new Matrix(1, dataMats.alpha.GetLength(0), "alpha");
        dataMatsEigen.beta = new Matrix(1, dataMats.beta.GetLength(0), "beta"); 
        dataMatsEigen.Q = new Matrix(dataMats.Q.GetLength(0), dataMats.Q.GetLength(1), "Q");
        dataMatsEigen.norms_mu_vec = new Matrix(1, dataMats.norms_mu_vec.GetLength(0), "norms_mu_vec");
        dataMatsEigen.U_T = new Matrix(dataMats.U_T.GetLength(0), dataMats.U_T.GetLength(1), "U_T");
        dataMatsEigen.Q_T = new Matrix(dataMats.Q_T.GetLength(0), dataMats.Q_T.GetLength(1), "Q_T");

        modelMatsEigen.fc1_weight = new Matrix(modelMats.fc1_weight.GetLength(0), modelMats.fc1_weight.GetLength(1), "fc1_weight");
        modelMatsEigen.fc1_bias = new Matrix(1, modelMats.fc1_bias.GetLength(0), "fc1_bias");
        modelMatsEigen.fc2_weight = new Matrix(modelMats.fc2_weight.GetLength(0), modelMats.fc2_weight.GetLength(1), "fc2_weight");
        modelMatsEigen.fc2_bias = new Matrix(1, modelMats.fc2_bias.GetLength(0), "fc2_bias");
        modelMatsEigen.fc3_weight = new Matrix(modelMats.fc3_weight.GetLength(0), modelMats.fc3_weight.GetLength(1), "fc3_weight");
        modelMatsEigen.fc3_bias = new Matrix(1, modelMats.fc3_bias.GetLength(0), "fc3_bias");
        modelMatsEigen.fc4_weight = new Matrix(modelMats.fc4_weight.GetLength(0), modelMats.fc4_weight.GetLength(1), "fc4_weight");
        modelMatsEigen.fc4_bias = new Matrix(1, modelMats.fc4_bias.GetLength(0), "fc4_bias");
        modelMatsEigen.fc5_weight = new Matrix(modelMats.fc5_weight.GetLength(0), modelMats.fc5_weight.GetLength(1), "fc5_weight");
        modelMatsEigen.fc5_bias = new Matrix(1, modelMats.fc5_bias.GetLength(0), "fc5_bias");
        modelMatsEigen.fc6_weight = new Matrix(modelMats.fc6_weight.GetLength(0), modelMats.fc6_weight.GetLength(1), "fc6_weight");
        modelMatsEigen.fc6_bias = new Matrix(1, modelMats.fc6_bias.GetLength(0), "fc6_bias");
        modelMatsEigen.fc7_weight = new Matrix(modelMats.fc7_weight.GetLength(0), modelMats.fc7_weight.GetLength(1), "fc7_weight");
        modelMatsEigen.fc7_bias = new Matrix(1, modelMats.fc7_bias.GetLength(0), "fc7_bias");
        modelMatsEigen.fc8_weight = new Matrix(modelMats.fc8_weight.GetLength(0), modelMats.fc8_weight.GetLength(1), "fc8_weight");
        modelMatsEigen.fc8_bias = new Matrix(1, modelMats.fc8_bias.GetLength(0), "fc8_bias");
        modelMatsEigen.fc9_weight = new Matrix(modelMats.fc9_weight.GetLength(0), modelMats.fc9_weight.GetLength(1), "fc9_weight");
        modelMatsEigen.fc9_bias = new Matrix(1, modelMats.fc9_bias.GetLength(0), "fc9_bias");
        modelMatsEigen.output_weight = new Matrix(modelMats.output_weight.GetLength(0), modelMats.output_weight.GetLength(1), "output_weight");
        modelMatsEigen.output_bias = new Matrix(1, modelMats.output_bias.GetLength(0), "output_bias");
    }

    public void SetEigenMatsValues()
    {
        dataMatsEigen.X_mu_vec = AssignMatValue1D(dataMatsEigen.X_mu_vec, dataMats.X_mu_vec);
        dataMatsEigen.Y_mu_vec = AssignMatValue1D(dataMatsEigen.Y_mu_vec, dataMats.Y_mu_vec);
        dataMatsEigen.U = AssignMatValue2D(dataMatsEigen.U, dataMats.U);
        dataMatsEigen.V = AssignMatValue2D(dataMatsEigen.V, dataMats.V);
        dataMatsEigen.alpha = AssignMatValue1D(dataMatsEigen.alpha, dataMats.alpha);
        dataMatsEigen.beta = AssignMatValue1D(dataMatsEigen.beta, dataMats.beta);
        dataMatsEigen.Q = AssignMatValue2D(dataMatsEigen.Q, dataMats.Q);
        dataMatsEigen.norms_mu_vec = AssignMatValue1D(dataMatsEigen.norms_mu_vec, dataMats.norms_mu_vec);
        dataMatsEigen.U_T = AssignMatValue2D(dataMatsEigen.U_T, dataMats.U_T);
        dataMatsEigen.Q_T = AssignMatValue2D(dataMatsEigen.Q_T, dataMats.Q_T);

        modelMatsEigen.fc1_weight = AssignMatValue2D(modelMatsEigen.fc1_weight, modelMats.fc1_weight);
        modelMatsEigen.fc1_bias = AssignMatValue1D(modelMatsEigen.fc1_bias, modelMats.fc1_bias);
        modelMatsEigen.fc2_weight = AssignMatValue2D(modelMatsEigen.fc2_weight, modelMats.fc2_weight);
        modelMatsEigen.fc2_bias = AssignMatValue1D(modelMatsEigen.fc2_bias, modelMats.fc2_bias);
        modelMatsEigen.fc3_weight = AssignMatValue2D(modelMatsEigen.fc3_weight, modelMats.fc3_weight);
        modelMatsEigen.fc3_bias = AssignMatValue1D(modelMatsEigen.fc3_bias, modelMats.fc3_bias);
        modelMatsEigen.fc4_weight = AssignMatValue2D(modelMatsEigen.fc4_weight, modelMats.fc4_weight);
        modelMatsEigen.fc4_bias = AssignMatValue1D(modelMatsEigen.fc4_bias, modelMats.fc4_bias);
        modelMatsEigen.fc5_weight = AssignMatValue2D(modelMatsEigen.fc5_weight, modelMats.fc5_weight);
        modelMatsEigen.fc5_bias = AssignMatValue1D(modelMatsEigen.fc5_bias, modelMats.fc5_bias);
        modelMatsEigen.fc6_weight = AssignMatValue2D(modelMatsEigen.fc6_weight, modelMats.fc6_weight);
        modelMatsEigen.fc6_bias = AssignMatValue1D(modelMatsEigen.fc6_bias, modelMats.fc6_bias);
        modelMatsEigen.fc7_weight = AssignMatValue2D(modelMatsEigen.fc7_weight, modelMats.fc7_weight);
        modelMatsEigen.fc7_bias = AssignMatValue1D(modelMatsEigen.fc7_bias, modelMats.fc7_bias);
        modelMatsEigen.fc8_weight = AssignMatValue2D(modelMatsEigen.fc8_weight, modelMats.fc8_weight);
        modelMatsEigen.fc8_bias = AssignMatValue1D(modelMatsEigen.fc8_bias, modelMats.fc8_bias);
        modelMatsEigen.fc9_weight = AssignMatValue2D(modelMatsEigen.fc9_weight, modelMats.fc9_weight);
        modelMatsEigen.fc9_bias = AssignMatValue1D(modelMatsEigen.fc9_bias, modelMats.fc9_bias);
        modelMatsEigen.output_weight = AssignMatValue2D(modelMatsEigen.output_weight, modelMats.output_weight);
        modelMatsEigen.output_bias = AssignMatValue1D(modelMatsEigen.output_bias, modelMats.output_bias);
    }

    public Matrix AssignMatValue1D(Matrix mat, float [] vec)
    {
        int row = 0;
        for (int col = 0; col < vec.GetLength(0); col++)
            mat.SetValue(row, col, vec[col]);

        return mat;
    }

    public Matrix AssignMatValue2D(Matrix mat, float [,] vec)
    {
        for (int row = 0; row < vec.GetLength(0); row++)
            for (int col = 0; col < vec.GetLength(1); col++)
                mat.SetValue(row, col, vec[row, col]);

        return mat;
    }
}
