# Talk to Repo

Talk to any GitHub repository and ask questions about the codebase. This app provides an interactive way to explore and understand and modify projects hosted on GitHub.

The app is hosted here: https://talk-to-repo.arjeo.com/. Instructions for hosting it yourself are provided below.

## Features

- Load any GitHub repository
- Input a GitHub API token for private repositories
- When loading the repo, the app checks locally for any secrets in files and discards sensitive files to avoid sending secrets to the AI
- Export the messages you see (currently to a new tab)
- Collect code that is relevant for changes you wish to do
- You can see the prompt used behind the scenes to generate the response
- Docker support for easy deployment

## Basic architecture

The app is a NextJS/Tailwind CSS frontend with a FastAPI backend. The backend uses a Pinecone vector DB on the free tier.

Given any GitHub repository, this app allows users to chat, fetch information from the repository, and display it in a user-friendly conversation interface.

## Architecture Overview

The Talk to Repo app consists of two main components:

### **Frontend**: A web interface built using Next.js and Tailwind CSS for a modern, responsive design. Users interact with this interface to input repository information, authenticate using their GitHub API tokens, and ask questions about the content of the repository.

   Key frontend components include:

   - `Header.js`: Displays the header with project title and repository link.
   - `ChatMessages.js`: Renders chat messages, including code snippets from the repository.
   - `GitHubInput.js`: Provides input fields for users to enter GitHub repository and API token information.

### **Backend**: A FastAPI server responsible for processing user input, fetching data from the GitHub repository, and returning relevant repository information as chat messages.

   Key backend components include:

#### `/load_repo`: (POST)

- Used to load repository information.
- Expected parameters:
  - `username`: The repository's owner's GitHub username (string)
  - `repo`: The name of the repository (string)
  - `token`: The GitHub API token for authentication (string). Pass an empty string for public repositories.

#### `/chat_stream`: (POST)

- Used to process chat requests and stream responses.
- Expected parameters:
  - `updatedMessages`: A list of chat message objects containing a `text` attribute (string) and a `role` attribute (either "system", "user", or "assistant").

Note that `/load_repo` and `/chat_stream` are POST requests, while `/fetch_CSV_data` is a GET request.

## Running with Docker Compose

The application now supports Docker Compose to simplify its deployment using both frontend and backend Docker containers. Before proceeding, ensure you have Docker and Docker Compose installed on your system.

1. Set up environment variables:

Copy the `.env.template` file to create a `.env` file for both the frontend and backend, if you haven't already:

```bash
    cp .env.template .env
    cd backend
    cp .env.template .env
    cd ..
```

Replace the placeholder values in both `.env` files with your actual API keys, organization ID, and other configuration values as needed. Save the files when you're done.

2. Build and run the Docker Compose:

To build and run the Talk to Repo app using Docker Compose, run the following command:
    
```bash
    docker-compose up --build
```


Docker Compose will then build both the frontend and backend Docker images and start the associated containers. Once the containers are up and running, you can access the app at `http://localhost:3000`.

To stop the Docker Compose, press `Ctrl+C` in the terminal, or run `docker-compose down` from another terminal.

## Running locally

Follow these steps to set up and run the Talk to Repo app locally.

1. Set up environment variables:

  Copy the .env.template file to create a .env file:

```bash
cp .env.template .env
cd backend
cp .env.template .env
```

 Replace the placeholder values with your actual API keys, organization ID, and other configuration values as needed. Save the file when you're done.

2. Install Node dependencies

```
npm i
```

3. Run the Node server

```
npm run dev
```

4. In another terminal, install the Python dependencies

```
# in the backend/ directory
cd backend 
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
```

6. Set up a Pinecone index. Give it a vector dimension of 1536 and name it `talk-to-repo`. 

7. Run the backend server

```
uvicorn main:app --reload
```

Now, you can access the app at `http://localhost:3000`.


## Potential improvements

- The dependency on Pinecone could be removed and replaced with a local vector database
- Replace the `chat_stream` endpoint with a WebSocket implementation to improve real-time communication.
- The splitter could be improved. Right now, it's a character splitter that favors newlines, but OpenAI has implemented a similar one that splits on tokens instead.
- Get inspired by Replit's Ghostwriter for the embeddings and retrieval mechanisms, to account for the hierarchy of code structure
- Use [Tree-Sitter](https://github.com/tree-sitter) to traverse the repo and create embedding at the function / module level
### CLI Support
Add the following CLI commands:

- `ttr load`: Load the current repo
- `ttr code1 "<a coding task>"`: Execute a provided coding task
- `ttr test "<a testing task>"`: Execute a provided testing task
- `ttr doc "<the purpose of the document>"`: Generate a document with a given purpose
- `ttr chat`: start a REPL

Stay tuned for further updates!


## Credit

Big thanks to [@mtenenholtz](https://github.com/mtenenholtz), who published [chat-twitter](https://twitter.com/marktenenholtz/status/1651568107192983553). This project started thanks to the embed-query-generate loop from that project.
