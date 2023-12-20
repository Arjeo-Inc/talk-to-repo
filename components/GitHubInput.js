import React, { useState, useEffect, useRef } from "react";
import { FaCheck } from 'react-icons/fa';


function GitHubInput(props) {
  const [username, setUsername] = useState("");
  const [repo, setRepo] = useState("");
  const [token, setToken] = useState("");
  const [loading, setLoading] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState(null);
  const [lastCommitHash, setLastCommitHash] = useState(null);
  const [hostingPlatform, setHostingPlatform] = useState("github");

  const fileInput = useRef(null);

  useEffect(() => {
    const storedHostingPlatform = localStorage.getItem("hostingPlatform");
    const storedUsername = localStorage.getItem("username");
    const storedRepo = localStorage.getItem("repo");
    const storedToken = localStorage.getItem("token");
  
    if (storedHostingPlatform) setHostingPlatform(storedHostingPlatform);
    if (storedUsername) setUsername(storedUsername);
    if (storedRepo) setRepo(storedRepo);
    if (storedToken) setToken(storedToken);
  }, []);

  const Spinner = () => (
    <div className="animate-spin w-5 h-5 border-t-2 border-white border-solid rounded-full align-items-center " />
  );
  
  const handleClick = async () => {
    setLoading(true); // Start loading
    setLoadingStatus(null);
    
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/load_repo`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        "Authorization": `Bearer ${process.env.NEXT_PUBLIC_TTR_API_KEY}`,
        body: JSON.stringify({ hostingPlatform, username, repo, token }), 
      });

      const data = await response.json();
      console.log(data);
      
      setLoading(false); // Stop loading
      setLoadingStatus(response.ok ? 'success' : 'error');
      if (response.ok) {
        localStorage.setItem("hostingPlatform", hostingPlatform);
        localStorage.setItem("username", username);
        localStorage.setItem("repo", repo);
        localStorage.setItem("token", token);
        props.clearMessages(); 
        setLastCommitHash(data.last_commit);
      }
    } catch (error) {
      console.error(error);
      setLoading(false); // Stop loading
      setLoadingStatus('error');
    }
  };

  const handleExportMessages = () => {
    const data = JSON.stringify(props.messages, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    window.open(url, '_blank');
  };

  const handleImportMessages = (event) => {
    const file = event.target.files[0];
    if (!file) return;
  
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const messages = JSON.parse(event.target.result);
        props.importMessages(messages);
      } catch (error) {
        console.error('Error importing messages:', error);
      }
    };
    reader.readAsText(file);
  };

  return (
    <div className="fixed top-0 left-0 h-screen w-48 bg-gray-700 text-white p-4">
      <h2 className="text-md font-medium mb-4">Load Repository</h2>
      <select
        className="mb-2 w-full text-sm p-1 border rounded-lg focus:outline-none focus:ring focus:border-blue-300 resize-none overflow-hidden bg-gray-600 text-gray-100"
        value={hostingPlatform}
        onChange={(e) => setHostingPlatform(e.target.value)}
      >
        <option value="github">GitHub</option>
        <option value="gitlab">GitLab</option>
        <option value="bitbucket">BitBucket</option>
        <option value="local">Local</option>
      </select>
      <input
        type="text"
        value={username}
        placeholder="GitHub Username"
        className="mb-2 w-full text-sm p-1 border rounded-lg focus:outline-none focus:ring focus:border-blue-300 resize-none overflow-hidden bg-gray-600 text-gray-100"        
        onChange={(e) => setUsername(e.target.value)}
      />
      <input
        type="text"
        value={repo}
        placeholder="Repository Name"
        className="mb-2 w-full text-sm p-1 border rounded-lg focus:outline-none focus:ring focus:border-blue-300 resize-none overflow-hidden bg-gray-600 text-gray-100"        
        onChange={(e) => setRepo(e.target.value)}
      />
      <input
        type="text"
        value={token}
        placeholder="Repository Access Token"
        className="mb-2 w-full text-sm p-1 border rounded-lg focus:outline-none focus:ring focus:border-blue-300 resize-none overflow-hidden bg-gray-600 text-gray-100"
        onChange={(e) => setToken(e.target.value)}
      />
   <button
     onClick={handleClick}
     className="w-full mb-2 text-sm px-1 py-1 rounded-lg bg-blue-500 text-white focus:outline-none hover:bg-blue-600 md:px-2 md:py-1 flex justify-center items-center"
   >
     {loading ? (
       <Spinner />
     ) : loadingStatus === 'success' ? (
       <FaCheck />
     ) : (
       'Load Repo'
     )}
   </button>
   <div className="mt-2 w-full">
     {lastCommitHash && (
       <div className="text-sm text-gray-200 p-1 bg-gray-600 text-gray-100 rounded-lg break-words w-full">
        Last commit hash: <span className="font-semibold">{lastCommitHash.slice(0, 10)}</span>
       </div>
      )}
    </div>
    <button
      onClick={handleExportMessages}
      className="w-full mb-2 text-sm px-1 py-1 rounded-lg bg-green-500 text-white focus:outline-none hover:bg-green-600 md:px-2 md:py-1 flex justify-center items-center"
    >
      Export Messages
    </button>
      <input
        type="file"
        accept=".json"
        style={{ display: 'none' }}
        onChange={handleImportMessages}
        ref={fileInput}
      />

      <button
        onClick={() => fileInput.current.click()}
        className="w-full mb-2 text-sm px-1 py-1 rounded-lg bg-green-500 text-white focus:outline-none hover:bg-green-600 md:px-2 md:py-1 flex justify-center items-center"
      >
        Import Messages
      </button>      
    </div>
  );
}

export default GitHubInput;