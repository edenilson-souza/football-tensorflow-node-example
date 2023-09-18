import * as tf from "@tensorflow/tfjs-node";
import { tidy, scalar, tensor2d, Tensor2D } from "@tensorflow/tfjs-node";

// Fazer previsões
// async function predictOutcome(newMatch: { homeScore: number; awayScore: number }) {
//     const xNew = tensor2d([[newMatch.homeScore, newMatch.awayScore]]);
//     const prediction = await tidy(() => model.predict(xNew) as Tensor2D);

//     const predictedResult = prediction.dataSync()[0] > 0.5 ? "Win" : "Loss";
//     console.log(`Resultado previsto para a nova partida: ${predictedResult}`);
// }
// // Treinar o modelo e fazer previsões
// trainModel().then(() => {
//     const newMatch = { homeScore: 25, awayScore: 20 };
//     predictOutcome(newMatch);
// });

// Carregar modelo e fazer previsões
tf.loadLayersModel("file://./model/model.json").then(loadedModel => {
    const newMatch = { homeScore: 22, awayScore: 21 };

    const xNew = tensor2d([[newMatch.homeScore, newMatch.awayScore]]);
    const prediction = tidy(() => loadedModel.predict(xNew) as Tensor2D);

    const predictedResult = prediction.dataSync()[0] > 0.5 ? "Win" : "Loss";
    console.log(`Resultado previsto para a nova partida: ${predictedResult}`);

    // predictOutcome(newMatch);
});
