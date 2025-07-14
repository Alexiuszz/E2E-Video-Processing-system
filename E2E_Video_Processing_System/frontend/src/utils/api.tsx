import axios from "axios";

export default axios.create({
  baseURL: "http://0.0.0.0:8081/", // or your FastAPI URL
});