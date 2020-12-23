using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System;
using UnityEngine.Profiling;


public class modelSimulation : MonoBehaviour
{
    const int width = 43;
    const int height = 22;
    int vertexCnt;
    int triangleCnt;
    int parseCnt = 0;
    List<float> vertexData = new List<float>();
    List<int> triangleData = new List<int>();
    Vector3[] Verts;
    Mesh mesh;
    DataMats dataMats;
    ModelMats modelMats;
    
    DataMatsEigen dataMatsEigen;
    ModelMatsEigen modelMatsEigen;
    Matrix flattenVertsEigen = new Matrix(1, 2838, "flattenVertsEigen");
    float [] flattenVerts = new float[2838];
    Matrix Zt_minus_1_Eigen = new Matrix(1, 128, "Zt_minus_1_Eigen");
    Matrix Zt_minus_2_Eigen = new Matrix(1, 128, "Zt_minus_2_Eigen");
    float [] Zt_minus_1 = new float[128];
    float [] Zt_minus_2 = new float[128];
    float [] Zpred = new float[128]; 
    Matrix ZpredEigen = new Matrix(1, 128, "ZpredEigen");
    Matrix verticesEigen = new Matrix(1, 128, "verticesEigen");
    Matrix normalsEigen = new Matrix(1, 128, "normalsEigen");
    float [] Zcorrection = new float[128];
    
    [Range(-Mathf.PI / 2f, Mathf.PI / 2f)]
    public float windRot = 0.006f;
    [Range(1000f, 10000f)]
    public float windStrength = 6000f;
    
    Matrix YtEigenOld = new Matrix(1, 2, "YtEigenOld");
    Matrix YtEigen = new Matrix(1, 2, "YtEigen");
    float [] Yt = new float[2]{0.006f, 6000};
    float [] Wt = new float[2];
    float[] product = new float[2838];
    float[] tempArray = new float[2838];
    float [] mathTempArray = new float[2838];
    Matrix inputEigen = new Matrix(1, 258, "inputEigen");
    Matrix productEigen = new Matrix(1, 1, "productEigen");
    
    void Start()
    {
        // Selection.activeGameObject = GameObject.Find("sim_model");
        // DestroyImmediate(Selection.activeGameObject);
        // Debug.Log("RUN HERE!");
        // Material material = Resources.Load("Custom_NewSurfaceShader", typeof(Material)) as Material;
        // mesh = gameObject.AddComponent<MeshFilter>().mesh;
        // MeshRenderer meshRenderer = gameObject.AddComponent<MeshRenderer>();
        // meshRenderer.material = material;
        Debug.Log("HERE");
        
        mesh = GetComponent<MeshFilter>().mesh;
           Verts = mesh.vertices;
        Debug.Log(Verts);
        TestLoad testload = new TestLoad();
        // Debug.Log(testload.dataMats.U[127, 11]);
        // Debug.Log(testload.modelMats.fc9_bias[3]);
        dataMats = testload.dataMats;
        modelMats = testload.modelMats;

        dataMatsEigen = testload.dataMatsEigen;
        modelMatsEigen = testload.modelMatsEigen;
        // Forward();
        test();
    }
    void GetInputs(int cnt)
    {
        int vertCnt = 0;
        
        // mesh.vertices
        foreach(Vector3 vert in vertices)
        {
            flattenVerts[vertCnt] = vert.x - dataMats.X_mu_vec[vertCnt];
            flattenVerts[vertCnt + 1] = vert.y - dataMats.X_mu_vec[vertCnt + 1];
            flattenVerts[vertCnt + 2] = vert.z - dataMats.X_mu_vec[vertCnt + 2];
            vertCnt = vertCnt + 3;
        }
        
        MatrixMult(flattenVerts, 2838, dataMats.U);
        Array.Copy(product, 0, Zt_minus_1, 0, 128);

        // Debug.Log("first part: ");
        // Debug.Log(string.Join(",", Zt_minus_1));

        ComputeInitModel(cnt);
        Array.Copy(initResult, 0, Zpred, 0, 128);

        // Yt[1] = 6000 + UnityEngine.Random.Range(-1000, 1000);
        Yt[1] = windStrength + UnityEngine.Random.Range(-1000, 1000);
        Yt[1] = Mathf.Clamp(Yt[1], 1000, 10000);
        // Yt[0] = 0.006f + UnityEngine.Random.Range(-2, 2) * (Mathf.PI / 18f);
        Yt[0] = Mathf.Clamp(windRot, -Mathf.PI / 2f, Mathf.PI / 2f);




        int vecLen = VectorSub(Yt, dataMats.Y_mu_vec);
        MatrixMult(mathTempArray, 2, dataMats.V);
        Array.Copy(product, 0, Wt, 0, 2);
        //flatten input array
        Array.Copy(Zpred, 0, product, 0, 128);
        Array.Copy(Zt_minus_1, 0, product, 128, 128);
        Array.Copy(Wt, 0, product, 256, 2);

        //test
        // float [] testInput = new float[258];
        // Array.Copy(product, 0, testInput, 0, 258);
        // Debug.Log("input: ");
        // Debug.Log(string.Join(",", testInput));
    } 

    
    void GetInputsEigen(int cnt)
    {
        int vertCnt = 0;
        
        // mesh.vertices
        // Debug.Log(vertices[0]);
        // foreach (Vector3 vert  in vertices)
        // {
        //     Debug.Log(vert);
        // }
        foreach(Vector3 vert in vertices)
        {
            flattenVertsEigen.SetValue(0, vertCnt, vert.x - dataMats.X_mu_vec[vertCnt]);
            flattenVertsEigen.SetValue(0, vertCnt + 1, vert.y - dataMats.X_mu_vec[vertCnt + 1]);
            flattenVertsEigen.SetValue(0, vertCnt + 2, vert.z - dataMats.X_mu_vec[vertCnt + 2]);
            Debug.Log(vert.x);
            vertCnt = vertCnt + 3;
        }
        Profiler.BeginSample("input compress");
        Zt_minus_1_Eigen = Matrix.Product(flattenVertsEigen, dataMatsEigen.U, Zt_minus_1_Eigen);
        Profiler.EndSample();
        // for (int i = 0; i < 128; i++)
        //     Zt_minus_1[i] = tempMat.GetValue(0, i);

        // Array.Copy(product, 0, Zt_minus_1, 0, 128);



        ComputeInitModelEigen(cnt);
        for (int i = 0; i < initEigen.GetCols(); i++)
            // Zpred[i] = initEigen.GetValue(0, i);
            ZpredEigen.SetValue(0, i, initEigen.GetValue(0, i));
        // Array.Copy(initResult, 0, Zpred, 0, 128);


        Yt[1] = windStrength + UnityEngine.Random.Range(-1000, 1000);
        Yt[1] = Mathf.Clamp(Yt[1], 1000, 10000);

        Yt[0] = Mathf.Clamp(windRot, -Mathf.PI / 2f, Mathf.PI / 2f);

        YtEigenOld.SetValue(0, 1, Yt[1]);
        YtEigenOld.SetValue(0, 0, Yt[0]);

        YtEigenOld = Matrix.Subtract(YtEigenOld, dataMatsEigen.Y_mu_vec, YtEigenOld);
        YtEigen = Matrix.Product(YtEigenOld, dataMatsEigen.V, YtEigen);

        for (int i = 0; i < 258; i++)
        {
            if (i < 128) inputEigen.SetValue(0, i, ZpredEigen.GetValue(0, i));
            else if (i >= 128 && i < 256) inputEigen.SetValue(0, i, Zt_minus_1_Eigen.GetValue(0, i - 128));
            else inputEigen.SetValue(0, i, YtEigen.GetValue(0, i - 256));
        }

    } 

    int cnt = 0;
    // Update is called once per frame
    void Update()
    {
        // Vector3[] vertices = mesh.vertices;
        // for (var i = 0; i < vertexCnt; i++)
        // {
        //     vertices[i] += Vector3.up * Time.deltaTime;
        // }

        // // assign the local vertices array into the vertices array of the Mesh.
        // mesh.vertices = vertices;
        // mesh.RecalculateBounds();   
        if (cnt % 6 == 0)
        {
            // ComputePipeline(cnt);
            ComputePipelineEigen(cnt);
            // Debug.Log("output: ");
            // Debug.Log(string.Join(",", Zcorrection));
            // Debug.Log("cnt: " + cnt);
        }
        cnt ++;

    }


    void Forward()
    {
        // for (int i = 0; i < 258; i++)
        //     product[i] = i;
        Profiler.BeginSample("forward");
        MatrixMultWithBiasReLU(product, 258, modelMats.fc1_weight, modelMats.fc1_bias, true);
        MatrixMultWithBiasReLU(product, 192, modelMats.fc2_weight, modelMats.fc2_bias, true);
        MatrixMultWithBiasReLU(product, 192, modelMats.fc3_weight, modelMats.fc3_bias, true);
        MatrixMultWithBiasReLU(product, 192, modelMats.fc4_weight, modelMats.fc4_bias, true);
        MatrixMultWithBiasReLU(product, 192, modelMats.fc5_weight, modelMats.fc5_bias, true);
        MatrixMultWithBiasReLU(product, 192, modelMats.fc6_weight, modelMats.fc6_bias, true);
        MatrixMultWithBiasReLU(product, 192, modelMats.fc7_weight, modelMats.fc7_bias, true);
        MatrixMultWithBiasReLU(product, 192, modelMats.fc8_weight, modelMats.fc8_bias, true);
        MatrixMultWithBiasReLU(product, 192, modelMats.fc9_weight, modelMats.fc9_bias, true);
        MatrixMultWithBiasReLU(product, 192, modelMats.output_weight, modelMats.output_bias, false);
        // Debug.Log(product[5]);
        Profiler.EndSample();
    }

    Matrix tempMat = new Matrix(1,192,"tempMat");
    Matrix layer1 = new Matrix(1,192,"layer1");
    Matrix layer2 = new Matrix(1,192,"layer2");
    Matrix layer3 = new Matrix(1,192,"layer3");
    Matrix layer4 = new Matrix(1,192,"layer4");
    Matrix layer5 = new Matrix(1,192,"layer5");
    Matrix layer6 = new Matrix(1,192,"layer6");
    Matrix layer7 = new Matrix(1,192,"layer7");
    Matrix layer8 = new Matrix(1,192,"layer8");
    Matrix layer9 = new Matrix(1,192,"layer9");
    Matrix output = new Matrix(1,128,"output");

    void ForwardEigen()
    {
        // Debug.Log(inputEigen.GetRows());
        // Debug.Log(inputEigen.GetCols());
        // Debug.Log(modelMatsEigen.fc1_weight.GetRows());
        // Debug.Log(modelMatsEigen.fc1_weight.GetCols());

        // Debug.Log("begin");

        // for (int i = 0; i < 258; i++)
        //     inputEigen.SetValue(0, i, i);

        // inputEigen.Print();
        // layer1 = Matrix.Product(inputEigen, modelMatsEigen.fc1_weight, layer1);
        // layer1 = Matrix.Add(layer1, modelMatsEigen.fc1_bias, layer1);
        // layer1 = Matrix.Relu(layer1, layer1);

        layer1 = Matrix.Layer(inputEigen, modelMatsEigen.fc1_weight, modelMatsEigen.fc1_bias, layer1);
        layer1 = Matrix.Relu(layer1, layer1);

        // layer2 = Matrix.Product(layer1, modelMatsEigen.fc2_weight, layer2);
        // layer2 = Matrix.Add(layer2, modelMatsEigen.fc2_bias, layer2);
        // layer2 = Matrix.Relu(layer2, layer2);

        layer2 = Matrix.Layer(layer1, modelMatsEigen.fc2_weight, modelMatsEigen.fc2_bias, layer2);
        layer2 = Matrix.Relu(layer2, layer2);

        // layer3 = Matrix.Product(layer2, modelMatsEigen.fc3_weight, layer3);
        // layer3 = Matrix.Add(layer3, modelMatsEigen.fc3_bias, layer3);
        // layer3 = Matrix.Relu(layer3, layer3);

        layer3 = Matrix.Layer(layer2, modelMatsEigen.fc3_weight, modelMatsEigen.fc3_bias, layer3);
        layer3 = Matrix.Relu(layer3, layer3);

        // layer4 = Matrix.Product(layer3, modelMatsEigen.fc4_weight, layer4);
        // layer4 = Matrix.Add(layer4, modelMatsEigen.fc4_bias, layer4);
        // layer4 = Matrix.Relu(layer4, layer4);

        layer4 = Matrix.Layer(layer3, modelMatsEigen.fc4_weight, modelMatsEigen.fc4_bias, layer4);
        layer4 = Matrix.Relu(layer4, layer4);

        // layer5 = Matrix.Product(layer4, modelMatsEigen.fc5_weight, layer5);
        // layer5 = Matrix.Add(layer5, modelMatsEigen.fc5_bias, layer5);
        // layer5 = Matrix.Relu(layer5, layer5);

        layer5 = Matrix.Layer(layer4, modelMatsEigen.fc5_weight, modelMatsEigen.fc5_bias, layer5);
        layer5 = Matrix.Relu(layer5, layer5);

        // layer6 = Matrix.Product(layer5, modelMatsEigen.fc6_weight, layer6);
        // layer6 = Matrix.Add(layer6, modelMatsEigen.fc6_bias, layer6);
        // layer6 = Matrix.Relu(layer6, layer6);

        layer6 = Matrix.Layer(layer5, modelMatsEigen.fc6_weight, modelMatsEigen.fc6_bias, layer6);
        layer6 = Matrix.Relu(layer6, layer6);

        // layer7 = Matrix.Product(layer6, modelMatsEigen.fc7_weight, layer7);
        // layer7 = Matrix.Add(layer7, modelMatsEigen.fc7_bias, layer7);
        // layer7 = Matrix.Relu(layer7, layer7);

        layer7 = Matrix.Layer(layer6, modelMatsEigen.fc7_weight, modelMatsEigen.fc7_bias, layer7);
        layer7 = Matrix.Relu(layer7, layer7);

        // layer8 = Matrix.Product(layer7, modelMatsEigen.fc8_weight, layer8);
        // layer8 = Matrix.Add(layer8, modelMatsEigen.fc8_bias, layer8);
        // layer8 = Matrix.Relu(layer8, layer8);
        layer8 = Matrix.Layer(layer7, modelMatsEigen.fc8_weight, modelMatsEigen.fc8_bias, layer8);
        layer8 = Matrix.Relu(layer8, layer8);

        // layer9 = Matrix.Product(layer8, modelMatsEigen.fc9_weight, layer9);
        // layer9 = Matrix.Add(layer9, modelMatsEigen.fc9_bias, layer9);
        // layer9 = Matrix.Relu(layer9, layer9);
        layer9 = Matrix.Layer(layer8, modelMatsEigen.fc9_weight, modelMatsEigen.fc9_bias, layer9);
        layer9 = Matrix.Relu(layer9, layer9);

        // output = Matrix.Product(layer9, modelMatsEigen.output_weight, output);
        // output = Matrix.Add(output, modelMatsEigen.output_bias, output);
        output = Matrix.Layer(layer9, modelMatsEigen.output_weight, modelMatsEigen.output_bias, output);

        // Debug.Log("end");

    }

    void ComputePipeline(int cnt)
    {
        Profiler.BeginSample("get inputs");
        GetInputs(cnt);
        Profiler.EndSample();
        
        Forward();
        
        Array.Copy(product, 0, Zcorrection, 0, 128);
        
        int vecLen = VectorAdd(Zpred, Zcorrection);
        Array.Copy(mathTempArray, 0, Zpred, 0, vecLen);

        Profiler.BeginSample("vertices pca decompression");
        MatrixMult(Zpred, 128, dataMats.U_T);
        Profiler.EndSample();
        
        Profiler.BeginSample("assign vertices");
        AssignVertices();
        Profiler.EndSample();
        
        //compute vertices' normals
        MatrixMult(Zpred, 128, dataMats.Q_T);
        
        AssignNormals();
        
        //update Zt_minus_1 and Zt_minus_2
        UpdateData();

    }

    void ComputePipelineEigen(int cnt)
    {
        Profiler.BeginSample("get inputs");
        GetInputsEigen(cnt);
        Profiler.EndSample();
        
        ForwardEigen();

        ZpredEigen = Matrix.Add(ZpredEigen, output, ZpredEigen);
        verticesEigen = Matrix.Product(ZpredEigen, dataMatsEigen.U_T, verticesEigen);

        
        Profiler.BeginSample("assign vertices");
        AssignVerticesEigen();
        Profiler.EndSample();
        
        //compute vertices' normals
        normalsEigen = Matrix.Product(ZpredEigen, dataMatsEigen.Q_T, normalsEigen); 
        AssignNormalsEigen();
        
        //update Zt_minus_1 and Zt_minus_2
        UpdateDataEigen();

    }

    int VectorElementWiseMult(float [] v1, float [] v2)
    {
        for (int i = 0; i < v1.GetLength(0); i++)
        {
            mathTempArray[i] = v1[i] * v2[i];
        }
        return v1.GetLength(0);        
    }

    int VectorAdd(float [] v1, float [] v2)
    {
        for (int i = 0; i < v1.GetLength(0); i++)
        {
            mathTempArray[i] = v1[i] + v2[i];
        }
        return v1.GetLength(0);
    }

    int VectorSub(float [] v1, float [] v2)
    {
        for (int i = 0; i < v1.GetLength(0); i++)
        {
            mathTempArray[i] = v1[i] - v2[i];
        }
        return v1.GetLength(0);
    }

    float [] initPart1 = new float[128];
    float [] initPart2 = new float[128];
    float [] initResult = new float[128];

    void ComputeInitModel(int cnt)
    {
        int vecLen = 0;
        if (cnt == 0)
            Array.Copy(Zt_minus_1, 0, Zt_minus_2, 0 ,128);

        // Matrix test_zt_minus1 = new Matrix(1, 128, "zt_minus1");
        // Matrix test_zt_minus2 = new Matrix(1, 128, "zt_minus2");
        // Matrix test_out = new Matrix(1, 128, "out");

        // for (int row = 0; row < 1; row++)
        // {
        //     for (int col = 0; col < 128; col++)
        //     {
        //         test_zt_minus1.SetValue(row, col, Zt_minus_1[col]);
        //         test_zt_minus2.SetValue(row, col, Zt_minus_2[col]);
        //     }
        // }

        vecLen = VectorSub(Zt_minus_1, Zt_minus_2);
        Array.Copy(mathTempArray, 0, initPart2, 0, vecLen);

        
        // test_out = Matrix.Subtract(test_zt_minus1, test_zt_minus2, test_out);
        // Debug.Log(string.Join(",", initPart2));

        // for (int row = 0; row < 1; row++)
        // {
        //     for (int col = 0; col < 128; col++)
        //     {
        //         Debug.Log(test_out.GetValue(row, col));
        //     }
        // }


        vecLen = VectorElementWiseMult(dataMats.beta, initPart2);
        Array.Copy(mathTempArray, 0, initPart2, 0, vecLen);

        vecLen = VectorElementWiseMult(dataMats.alpha, Zt_minus_1);
        Array.Copy(mathTempArray, 0, initPart1, 0, vecLen);

        vecLen = VectorAdd(initPart1, initPart2);
        Array.Copy(mathTempArray, 0, initResult, 0, vecLen);

        // if (cnt == 0) 
        // {
        //     Debug.Log("Zt minus 1: ");
        //     Debug.Log(string.Join(",", Zt_minus_1));
        //     Debug.Log("Zt minus 2: ");
        //     Debug.Log(string.Join(",", Zt_minus_2));
        //     Debug.Log("init haha  part: ");
        //     Debug.Log(string.Join(",", initResult));
        // }


        // Debug.Log("second part: ");
        // Debug.Log(string.Join(",", initPart2));
        
        // return VectorAdd(VectorElementWiseMult(dataMats.alpha, Zt_minus_1), VectorElementWiseMult(dataMats.beta, VectorSub(Zt_minus_1, Zt_minus_2)));
    }

    Matrix initPart2Eigen = new Matrix(1,1,"initPart1Eigen");
    Matrix initEigen = new Matrix(1,1,"initEigen");
    void ComputeInitModelEigen(int cnt)
    {
        if(cnt == 0)
        {
            for (int col = 0; col < Zt_minus_1_Eigen.GetCols(); col++)
                Zt_minus_2_Eigen.SetValue(0, col, Zt_minus_1_Eigen.GetValue(0, col));
        }

        initPart2Eigen = Matrix.Subtract(Zt_minus_1_Eigen, Zt_minus_2_Eigen, initPart2Eigen);
        initPart2Eigen = Matrix.PointwiseProduct(dataMatsEigen.beta, initPart2Eigen, initPart2Eigen);

        initEigen = Matrix.PointwiseProduct(dataMatsEigen.alpha, Zt_minus_1_Eigen, initEigen);
        initEigen = Matrix.Add(initPart2Eigen, initEigen, initEigen);

        // if(cnt == 0)
        // {
        //     Debug.Log("Zt_minus_1_Eigen");
        //     Zt_minus_1_Eigen.Print();
        //     Debug.Log("Zt_minus_2_Eigen");
        //     Zt_minus_2_Eigen.Print();
        //     Debug.Log("init part");
        //     initEigen.Print();
        // }

    }


    /// This method takes 3 matrices and one boolean variable as parameters
    /// (X is with dimension 1*m, weight matrix is with dimension m*n, bias is with dimension 1*n) 
    /// Compute ReLU(MatrixMult(X, weight) + bias) if bool ifActivate is set to true, 
    /// Otherwise, compute (MatrixMult(X, weight) + bias), 
    /// notice that for output layer, ifActivate should be set to false,
    /// the return is a matrix with dimension n*1,
    /// this computation is part of Neural Network's forward computation process
    void MatrixMultWithBiasReLU(float[] X, int XLen, float[,] weight, float[] bias, bool ifActivate)
    {
        if (XLen != weight.GetLength(0) || weight.GetLength(1) != bias.Length)
        {
            Debug.Log("XLen: " + XLen);
            Debug.Log("weight0: " + weight.GetLength(0));
            Debug.Log("weight1: " + weight.GetLength(1));
            Debug.Log("bias.Length: " + bias.Length);
            Debug.Log("Math error, please check parameters' dimensions");
        }
        else
        {

            /*
            During each loop, matrix multiplication is computed firstly,
            then bias is added to previous result,
            finally, ReLU activation function is applied.
            i in outer loop is the index of weight's column number,
            j in inner loop is the index of X's column number
            */
            Array.Copy(X, 0, tempArray, 0, XLen);
            for (int i = 0; i < weight.GetLength(1); i++)
            {
                float tempVal = 0;
                for (int j = 0; j < XLen; j++)
                {
                    tempVal += (tempArray[j] * weight[j, i]);
                }
                tempVal += bias[i];
                if (ifActivate)
                    product[i] = Mathf.Max(tempVal, 0);
                else
                    product[i] = tempVal;
            }
        }
    }
    void MatrixMult(float[] X, int XLen, float[,] weight)
    {
        if (XLen != weight.GetLength(0))
        {
            Debug.Log("XLen: " + XLen);
            Debug.Log("weight0: " + weight.GetLength(0));
            Debug.Log("weight1: " + weight.GetLength(1));
            Debug.Log("Math error, please check parameters' dimensions");
        }
        else
        {

            /*
            During each loop, matrix multiplication is computed firstly,

            i in outer loop is the index of weight's column number,
            j in inner loop is the index of X's column number
            */
            Array.Copy(X, 0, tempArray, 0, XLen);
            for (int i = 0; i < weight.GetLength(1); i++)
            {
                float tempVal = 0;
                for (int j = 0; j < XLen; j++)
                {
                    tempVal += (tempArray[j] * weight[j, i]);
                }

                product[i] = tempVal;
            }
        }
    }
    Vector3[] vertices = new Vector3[946];
    void AssignVertices()
    {
        // Profiler.BeginSample("assign vertex");
        // vertices = mesh.vertices;
        // Profiler.EndSample();
        // Debug.Log(vertexCnt);
        
        for (int i = 0; i < vertexCnt; i++)
        {
            // Debug.Log(i);
            // Debug.Log("debug: " + i * 3);
            vertices[i].x = product[i * 3] + dataMats.X_mu_vec[i * 3];
            vertices[i].y = product[i * 3 + 1] + dataMats.X_mu_vec[i * 3 + 1];
            vertices[i].z = product[i * 3 + 2] + dataMats.X_mu_vec[i * 3 + 2];
            // Debug.Log(vertices[i].x);
            // Debug.Log(vertices[i].y);
            // Debug.Log(vertices[i].z);
        }
        

        // assign the local vertices array into the vertices array of the Mesh.
        mesh.vertices = vertices;

        mesh.RecalculateBounds();
    }

    void AssignVerticesEigen()
    {
        for (int i = 0; i < vertexCnt; i++)
        {
            vertices[i].x = verticesEigen.GetValue(0, i * 3) + dataMats.X_mu_vec[i * 3];
            vertices[i].y = verticesEigen.GetValue(0, i * 3 + 1) + dataMats.X_mu_vec[i * 3 + 1];
            vertices[i].z = verticesEigen.GetValue(0, i * 3 + 2) + dataMats.X_mu_vec[i * 3 + 2];
        }
        

        // assign the local vertices array into the vertices array of the Mesh.
        mesh.vertices = vertices;
        mesh.RecalculateBounds();        
    }

    Vector3[] normals = new Vector3[946];
    void AssignNormals()
    {
        // normals = mesh.normals;
        // Profiler.BeginSample("assign normals");
        for (var i = 0; i < vertexCnt; i++)
        {
            normals[i].x = product[i * 3] + dataMats.norms_mu_vec[i * 3];
            normals[i].y = product[i * 3 + 1] + dataMats.norms_mu_vec[i * 3 + 1];
            normals[i].z = product[i * 3 + 2] + dataMats.norms_mu_vec[i * 3 + 2];
        }
        // Profiler.EndSample();

        // assign the local vertices array into the vertices array of the Mesh.
        mesh.normals = normals;
        mesh.RecalculateBounds();
    }

    void AssignNormalsEigen()
    {
        for (var i = 0; i < vertexCnt; i++)
        {
            normals[i].x = normalsEigen.GetValue(0, i * 3) + dataMats.norms_mu_vec[i * 3];
            normals[i].y = normalsEigen.GetValue(0, i * 3 + 1) + dataMats.norms_mu_vec[i * 3 + 1];
            normals[i].z = normalsEigen.GetValue(0, i * 3 + 2) + dataMats.norms_mu_vec[i * 3 + 2];
        }
        // Profiler.EndSample();

        // assign the local vertices array into the vertices array of the Mesh.
        mesh.normals = normals;
        mesh.RecalculateBounds();
    }

    void UpdateData()
    {
        // Vector3[] vertices = mesh.vertices;

        for (int i = 0; i < 128; i++)
        {
            // Debug.Log(i);
            // Debug.Log("debug: " + i * 3);
            Zt_minus_2[i] = Zt_minus_1[i];
            Zt_minus_1[i] = Zpred[i];

        }

    }

    void UpdateDataEigen()
    {
        // Vector3[] vertices = mesh.vertices;

        for (int i = 0; i < 128; i++)
        {
            // Debug.Log(i);
            // Debug.Log("debug: " + i * 3);
            Zt_minus_2_Eigen.SetValue(0, i, Zt_minus_1_Eigen.GetValue(0, i));
            Zt_minus_1_Eigen.SetValue(0, i, ZpredEigen.GetValue(0, i));

        }

    }

    void test()
    {
        float [] test1 = new float[2]{1,5};
        float [] test2 = new float[2]{6,9};
        // Debug.Log("math test");
        // Debug.Log(string.Join(",", VectorSub(test1, test2)));
        // Debug.Log(string.Join(",", VectorAdd(test1, test2)));
        // Debug.Log(string.Join(",", VectorElementWiseMult(test1, test2)));
    }

}



// public class modelSim : MonoBehaviour
// {
//     public float width = 1;
//     public float height = 1;

//     public void Start()
//     {
//         MeshRenderer meshRenderer = gameObject.AddComponent<MeshRenderer>();
//         meshRenderer.sharedMaterial = new Material(Shader.Find("Standard"));

//         MeshFilter meshFilter = gameObject.AddComponent<MeshFilter>();

//         Mesh mesh = new Mesh();

//         Vector3[] vertices = new Vector3[4]
//         {
//             new Vector3(0, 0, 0),
//             new Vector3(width, 0, 0),
//             new Vector3(0, height, 0),
//             new Vector3(width, height, 0)
//         };
//         mesh.vertices = vertices;

//         int[] tris = new int[6]
//         {
//             // lower left triangle
//             0, 2, 1,
//             // upper right triangle
//             2, 3, 1
//         };
//         mesh.triangles = tris;

//         Vector3[] normals = new Vector3[4]
//         {
//             -Vector3.forward,
//             -Vector3.forward,
//             -Vector3.forward,
//             -Vector3.forward
//         };
//         mesh.normals = normals;

//         Vector2[] uv = new Vector2[4]
//         {
//             new Vector2(0, 0),
//             new Vector2(1, 0),
//             new Vector2(0, 1),
//             new Vector2(1, 1)
//         };
//         mesh.uv = uv;

//         meshFilter.mesh = mesh;
//     }
// }

