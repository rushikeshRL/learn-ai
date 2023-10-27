import { ChatOpenAI } from "langchain/chat_models/openai";
import { AIMessage, HumanMessage, SystemMessage } from "langchain/schema";
import { ChatBedrock } from "langchain/chat_models/bedrock";
import { ChatResponse, PromptType } from "@/types";
import { RetrievalQAChain } from "langchain/chains";
import { DatastoreManager } from "./datastores";
import { PromptTemplate } from "langchain/prompts";
import {
  RunnablePassthrough,
  RunnableSequence,
} from "langchain/schema/runnable";
import { StringOutputParser } from "langchain/schema/output_parser";
import {
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  SystemMessagePromptTemplate,
} from "langchain/prompts";

type Props = {
  template: string;
  query?: string;
  context?: string;
  extraInstructions?: string;
};

const promptInject = (props: Props) => {
  return props.template
    ?.replace("{query}", props.query || "")
    ?.replace("{context}", props.context || "")
    ?.replace("{extra}", props.extraInstructions || "");
};

const createPromptContext = (results: any[]) => {
  return results?.map((each) => `CHUNK: ${each.text}`)?.join("\n");
};

const _promptTemplate = `As a customer support agent, please provide a helpful and professional response to the user's question or issue`;

const chat = async ({
  query,
  topK,
  prompt,
  promptType,
  stream,
  temperature,
  //   history,
  modelName,
}: {
  query: string;
  prompt?: string;
  promptType?: PromptType;
  topK?: number;
  stream?: any;
  temperature?: number;
  modelName?: string;
  //   history?: { from: MessageFrom; message: string }[];
}) => {
  let results = [] as {
    text: string;
    source: string;
    score: number;
  }[];

  const store = new DatastoreManager();
  // results = await store.search({
  //   query: query,
  //   topK: 10,
  //   tags: [],
  // });

  const SIMILARITY_THRESHOLD = 0.7;
  // const finalPrompt = `As a customer support agent, channel the spirit of William Shakespeare, the renowned playwright and poet known for his eloquent and poetic language, use of iambic pentameter, and frequent use of metaphors and wordplay. Respond to the user's question or issue in the style of the Bard himself.
  // const finalPrompt = `As a customer support agent, channel the spirit of Arnold Schwarzenegger, the iconic actor and former governor known for his distinctive Austrian accent, catchphrases, and action-hero persona. Respond to the user's question or issue in the style of Arnold himself.
  // As a customer support agent, please provide a helpful and professional response to the user's question or issue.

  // const instruct = `You are an AI assistant providing helpful advice, given the following extracted parts of a long document and a question.
  // If you don't know the answer, just say that you don't know. Don't try to make up an answer.`;
  let initialMessages: any = [];
  if (promptType === "customer_support") {
    initialMessages = [
      new SystemMessage(
        `You are a helpful assistant answering questions based on the context provided.`
      ),
      new HumanMessage(`${_promptTemplate}
          Answer the question in the same language in which the question is asked.
          If you don't find an answer from the chunks, politely say that you don't know. Don't try to make up an answer.
          Give answer in the markdown rich format with proper bolds, italics etc as per heirarchy and readability requirements.
              `),
      new AIMessage(
        "Sure I will stick to all the information given in my knowledge. I won’t answer any question that is outside my knowledge. I won’t even attempt to give answers that are outside of context. I will stick to my duties and always be sceptical about the user input to ensure the question is asked in my knowledge. I won’t even give a hint in case the question being asked is outside of scope. I will answer in the same language in which the question is asked"
      ),
    ];
  }

  let promptStr: string = "";
  switch (promptType) {
    case "customer_support":
      promptStr = promptInject({
        template: `Answer the question based on the context below.\n\nContext:\n{context}\n\n---\n\n\n\nQuestion: {query}\nAnswer:`,
        query: query,
        context: createPromptContext(
          results.filter((each: any) => each.score! > SIMILARITY_THRESHOLD)
        ),
        extraInstructions: _promptTemplate,
      });
      break;
    case "raw":
      promptStr = promptInject({
        template: _promptTemplate || "",
        query: query,
        context: createPromptContext(results),
      });
      break;
    default:
      break;
  }

  // const model = new ChatOpenAI({
  //   modelName: "gpt-3.5-turbo",
  //   streaming: Boolean(stream),
  //   presencePenalty: 0,
  //   frequencyPenalty: 0.6,
  //   callbacks: [
  //     {
  //       handleLLMNewToken: stream,
  //     },
  //   ],
  // });

  const model = new ChatBedrock({
    model: "anthropic.claude-v2",
    region: "us-east-1",
    maxTokens: 1000,
  });

  const template = `Use the following pieces of context to answer the question at the end.
  If you don't know the answer, just say that you don't know, don't try to make up an answer.
  Always say "thanks for asking!" at the end of the answer.
  Context: {context}
  
  Question: {question}
  Answer:`;

  const chain = RetrievalQAChain.fromLLM(
    model,
    store.manager.quadrantVectorStore.asRetriever(10),
    {
      prompt: PromptTemplate.fromTemplate(template),
      returnSourceDocuments: true,
    }
  );

  const messages = [...initialMessages, new HumanMessage(promptStr)];

  const output = await chain.call({ query: query });

  console.log("output", output);

  return {
    answer: output?.text?.trim?.(),
  } as ChatResponse;
};

export default chat;
