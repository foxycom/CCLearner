package vt.cs;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Collections;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.util.Properties;
import java.util.concurrent.TimeUnit;

public class CCLearner_Train {

  public static String config_file = "C:/Users/timvs/CCLearner/CCLearner.conf";

  public static String training_file;

  public static String output_dir;

  public static int seed;
  public static double learningRate;
  public static int batchSize;
  public static int nEpochs;

  public static int numInputs;
  public static int numOutputs;
  public static int numHiddenNodes;

  public static void loadConfig() {
    try {
      Properties prop = new Properties();
      InputStream is = new FileInputStream(config_file);

      prop.load(is);

      output_dir = prop.getProperty("output.dir");

      training_file = prop.getProperty("feature.file.path");

      seed = Integer.valueOf(prop.getProperty("training.seed"));
      learningRate = Double.valueOf(prop.getProperty("training.learningRate"));
      batchSize = Integer.valueOf(prop.getProperty("training.batchSize"));
      nEpochs = Integer.valueOf(prop.getProperty("training.iteration"));

      numInputs = Integer.valueOf(prop.getProperty("training.input.num"));
      numOutputs = Integer.valueOf(prop.getProperty("training.output.num"));
      numHiddenNodes = Integer.valueOf(prop.getProperty("training.hidden.num"));
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  public static void main(String[] args) throws Exception {

    loadConfig();

    //Load the training data:
    RecordReader rr = new CSVRecordReader();
    rr.initialize(new FileSplit(new File(training_file)));
    var trainIter = new RecordReaderDataSetIterator(rr, batchSize, 0, 2);

    long start = System.nanoTime();
    var builder = new NeuralNetConfiguration.Builder()
        .seed(seed)
        //.iterations(1)
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .updater(new Nesterovs(learningRate, 0.9))
        //.learningRate(learningRate)
        //.updater(Updater.NESTEROVS).momentum(0.9)
        .list()
        .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
            .weightInit(WeightInit.XAVIER)
            //.activation("relu")
            .activation(Activation.RELU)
            .build())
        .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
            .weightInit(WeightInit.XAVIER)
            //.activation("relu")
            .activation(Activation.RELU)
            .build())
        /*
        .layer(2, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
            .weightInit(WeightInit.XAVIER)
            .activation("relu")
            .build())
        .layer(3, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
            .weightInit(WeightInit.XAVIER)
            .activation("relu")
            .build())
        .layer(4, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
            .weightInit(WeightInit.XAVIER)
            .activation("relu")
            .build())
        .layer(5, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
            .weightInit(WeightInit.XAVIER)
            .activation("relu")
            .build())
        .layer(6, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
            .weightInit(WeightInit.XAVIER)
            .activation("relu")
            .build())
        .layer(7, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
            .weightInit(WeightInit.XAVIER)
            .activation("relu")
            .build())
        /*
        .layer(8, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
            .weightInit(WeightInit.XAVIER)
            .activation("relu")
            .build())
        .layer(9, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
            .weightInit(WeightInit.XAVIER)
            .activation("relu")
            .build())
        */
        .layer(2, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
            .weightInit(WeightInit.XAVIER)
            .activation(Activation.SOFTMAX)
            .nIn(numHiddenNodes)
            .nOut(numOutputs).build());
    //.pretrain(false)
    //.backprop(true)
    builder.setBackpropType(BackpropType.Standard);
    MultiLayerConfiguration conf = builder.build();

    MultiLayerNetwork model = new MultiLayerNetwork(conf);
    model.init();
    model.setListeners(Collections.singletonList(new ScoreIterationListener(10)));

    for (int n = 0; n < nEpochs; n++) {
      model.fit(trainIter);
    }

    File model_File = new File(output_dir + "model.mdl");
    ModelSerializer.writeModel(model, model_File, true);

    long end = System.nanoTime();

    System.out.println("Time Cost: " + TimeUnit.NANOSECONDS.toMillis(end - start) + "ms");
  }
}