import asyncio
import os
import subprocess
import time
from playwright.async_api import async_playwright

async def run():
    # Start the server
    server_process = subprocess.Popen(["uvicorn", "dashboard.app:app", "--port", "8000"])
    
    # Wait for server to start
    time.sleep(2)
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1280, "height": 800})
        
        # Intercept the /evaluate API call to return mock data for consistent screenshots
        async def handle_route(route):
            if "/evaluate" in route.request.url and "batch" not in route.request.url:
                await route.fulfill(
                    status=200,
                    content_type="application/json",
                    body='{"scores": {"faithfulness": {"score": 0.95, "explanation": "Excellent faithfulness.", "details": {}}, "relevance": {"score": 0.88, "explanation": "Highly relevant.", "details": {}}, "hallucination": {"score": 0.98, "explanation": "No hallucination detected.", "details": {}}, "coherence": {"score": 0.92, "explanation": "Very coherent.", "details": {}}}, "overall": 0.93}'
                )
            else:
                await route.continue_()
                
        await page.route("**/*", handle_route)
        
        await page.goto("http://localhost:8000")
        
        # Fill in the form
        await page.fill("#input-question", "What is RAG?")
        await page.fill("#input-context", "Retrieval-Augmented Generation (RAG) is the process of optimizing the output of a large language model, so it references an authoritative knowledge base outside of its training data sources before generating a response.")
        await page.fill("#input-answer", "RAG stands for Retrieval-Augmented Generation. It improves LLMs by retrieving information from an external knowledge base to provide more accurate and grounded responses.")
        
        # Click evaluate
        await page.click("#btn-evaluate")
        
        # Wait for rings to animate
        await page.wait_for_timeout(2000)
        
        # Take screenshot
        await page.screenshot(path="dashboard_screenshot.png")
        
        await browser.close()
        
    # Stop the server
    server_process.terminate()
    server_process.wait()
    print("Screenshot saved to dashboard_screenshot.png")

if __name__ == "__main__":
    asyncio.run(run())
