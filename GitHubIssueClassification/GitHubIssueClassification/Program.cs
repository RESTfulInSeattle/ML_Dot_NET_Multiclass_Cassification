using System;
using System.IO;
using System.Linq;
using Microsoft.ML;

//From:  https://docs.microsoft.com/en-us/dotnet/machine-learning/tutorials/github-issue-classification

namespace GitHubIssueClassification
{
    class Program
    {
        private static string _appPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string _trainDataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "issues_train.tsv");
        private static string _testDataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "issues_test.tsv");
        private static string _modelPath => Path.Combine(_appPath, "..", "..", "..", "Models", "model.zip");

        private static MLContext _mlContext;
        private static PredictionEngine<GitHubIssue, IssuePrediction> _predEngine;
        private static ITransformer _trainedModel;
        static IDataView _trainingDataView;

        static void Main(string[] args)
        {
            _mlContext = new MLContext(seed: 0);
            _trainingDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue>(_trainDataPath, hasHeader: true);

            var pipeline = ProcessData();

            var trainingPipeline = BuildAndTrainModel(_trainingDataView, pipeline);

            Evaluate(_trainingDataView.Schema);

            PredictIssue();
        }

        public static IEstimator<ITransformer> BuildAndTrainModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
        {
            var trainingPipeline = pipeline.Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            _trainedModel = trainingPipeline.Fit(trainingDataView);

            _predEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(_trainedModel);

            GitHubIssue issue = new GitHubIssue()
            {
                Title = "WebSockets communication is slow in my machine",
                Description = "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.."
            };

            var prediction = _predEngine.Predict(issue);

            Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.Area} ===============");
            
            return trainingPipeline;
        }

        public static void Evaluate(DataViewSchema trainingDataViewSchema)
        {
            var testDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue>(_testDataPath, hasHeader: true);

            var testMetrics = _mlContext.MulticlassClassification.Evaluate(_trainedModel.Transform(testDataView));

            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
            Console.WriteLine($"*       MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");
            Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
            Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
            Console.WriteLine($"*************************************************************************************************************");

            SaveModelAsFile(_mlContext, trainingDataViewSchema, _trainedModel);
        }

        private static void PredictIssue()
        {
            ITransformer loadedModel = _mlContext.Model.Load(_modelPath, out var modelInputSchema);

            GitHubIssue singleIssue = new GitHubIssue() { Title = "Entity Framework crashes", Description = "When connecting to the database, EF is crashing" };

            //Not thread-safe, for thread safety, use PredictionEnginePool
            //https://docs.microsoft.com/en-us/dotnet/machine-learning/how-to-guides/serve-model-web-api-ml-net#register-predictionenginepool-for-use-in-the-application
            _predEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(loadedModel);

            var prediction = _predEngine.Predict(singleIssue);
            Console.WriteLine($"=============== Single Prediction - Result: {prediction.Area} ===============");



        }

        private static void SaveModelAsFile(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model)
        {
            mlContext.Model.Save(model, trainingDataViewSchema, _modelPath);
        }

        public static IEstimator<ITransformer> ProcessData()
        {
            //Don't use caching for large datasets, but with medium and small, it speeds up training time
            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Area", outputColumnName: "Label")
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Title", outputColumnName: "TitleFeaturized"))
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Description", outputColumnName: "DescriptionFeaturized"))
                .Append(_mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
                .AppendCacheCheckpoint(_mlContext);

            return pipeline;
        }
    }
}
