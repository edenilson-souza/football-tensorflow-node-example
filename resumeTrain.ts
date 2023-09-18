import * as tf from "@tensorflow/tfjs-node";
import { tensor2d } from "@tensorflow/tfjs-node";

async function retrainModel() {
    // Carregar o modelo previamente salvo a partir do arquivo
    const model = await tf.loadLayersModel("file://./model/model.json");

    // Novos dados para treinamento
    const matches = [
        { homeScore: 32, awayScore: 30, result: "Win" }, //Adicione mais estatísticas aqui...
        { homeScore: 30, awayScore: 32, result: "Loss" },
        { homeScore: 30, awayScore: 30, result: "Loss" },
        { homeScore: 30, awayScore: 40, result: "Loss" },
        { homeScore: 20, awayScore: 15, result: "Win" },
        { homeScore: 15, awayScore: 20, result: "Loss" },
        { homeScore: 20, awayScore: 20, result: "Loss" },
        { homeScore: 26, awayScore: 25, result: "Win" }
        // Adicione mais partidas aqui...
    ];

    // Preparando os dados
    const inputData: number[][] = [];
    const outputData: number[] = [];

    for (const match of matches) {
        const homeScore = match.homeScore;
        const awayScore = match.awayScore;
        //Preparando os dados de entrada e saída
        //Se adicionar mais estatísticas, adicione mais valores aqui
        const result = match.result === "Win" ? 1 : 0; // Codificar 'Win' como 1 e 'Loss' como 0

        inputData.push([homeScore, awayScore]); // Adicionando os dados de entrada
        outputData.push(result); // Adicionando os dados de saída
    }

    // Preparando os novos dados da mesma forma que no exemplo anterior
    const xData = tensor2d(inputData);
    const yData = tensor2d(outputData, [outputData.length, 1]);

    model.compile({ optimizer: tf.train.adam(), loss: "binaryCrossentropy", metrics: ["accuracy"] });

    // Continuando o treinamento
    async function continueTraining() {
        const epochsTotal = 10000;
        const epochsPerBatch = 10;
        const batchSize = 4;
        const verbose = 1;

        await model
            .fit(xData, yData, {
                epochs: epochsTotal,
                verbose: verbose,
                batchSize: batchSize,
                stepsPerEpoch: Math.ceil(xData.shape[0] / batchSize) * epochsPerBatch,
                shuffle: true,
                callbacks: {
                    onEpochEnd: (epoch: any, logs: any) => {
                        console.log(`Epoch ${epoch + 1}/${epochsTotal} - loss: ${logs?.loss.toFixed(4)} - acc: ${logs?.acc.toFixed(4)}`);
                    }
                }
            })
            .then(() => {});

        console.log("Treinamento concluído.");
        model.summary();

        // Salvar o modelo atualizado se desejar
        const trainDate = new Date();
        await model.save(`file://./${trainDate.getTime()}`).then(() => {
            console.log("Modelo salvo.");
        });
    }

    // Continuar o treinamento do modelo com os novos dados
    continueTraining().then(() => {
        console.log("Treinamento concluído com os novos dados.");
    });
}

retrainModel();
