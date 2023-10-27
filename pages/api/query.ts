import { NextApiResponse, NextApiRequest } from "next";

import { ChatRequest } from "@/types";
import AgentManager from "@/utils/agent";
import { NextResponse } from "next/server";

const handler = async (req: NextApiRequest, res: NextApiResponse) => {
  const data = req.body as ChatRequest;

  const manager = new AgentManager({ topK: 5 });

  const answer = await manager.query({
    input: data.query,
    // stream: data.streaming ? streamData : undefined,
  });

  console.log("answer is ", answer);

  res.status(200).json(answer);

  // return new Response(answer, {
  //   status: 200,
  // });

  // return {
  //   answer,
  // };
};

export default handler;
