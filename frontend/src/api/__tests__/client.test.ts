
import { describe, it, expect, vi, afterEach } from "vitest";
import { sendAssistantMessage } from "../client";

describe("Assistant API error handling", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("converts FastAPI validation errors into readable messages", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            detail: [
              {
                loc: ["body", "question"],
                msg: "Field required",
                type: "missing",
              },
            ],
          }),
          {
            status: 422,
            headers: { "Content-Type": "application/json" },
          }
        )
      )
    );

    await expect(
      sendAssistantMessage("")
    ).rejects.toThrow("question: Field required");
  });

  it("displays backend string error messages", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            detail: "The AI Assistant is unavailable.",
          }),
          { status: 503 }
        )
      )
    );

    await expect(
      sendAssistantMessage("What is NDVI?")
    ).rejects.toThrow("The AI Assistant is unavailable.");
  });

  it("uses the default message for non-JSON errors", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response("Internal Server Error", {
          status: 500,
        })
      )
    );

    await expect(
      sendAssistantMessage("What is NDVI?")
    ).rejects.toThrow(
      "Failed to get a response from the AI Assistant."
    );
  });

    it("displays a readable message when the backend is unreachable", async () => {
    vi.stubGlobal(
        "fetch",
        vi.fn().mockRejectedValue(
        new TypeError("Failed to fetch")
        )
    );

    await expect(
        sendAssistantMessage("What is NDVI?")
    ).rejects.toThrow(
        "Unable to connect to the AI Assistant. Please check your internet connection and make sure the backend server is running."
    );
    });

});
