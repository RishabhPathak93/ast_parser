import os
import glob
import time
import re
import logging
import warnings
from pymongo import MongoClient
import uuid
from datetime import datetime
from bson import ObjectId
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
import hashlib
import psutil
import asyncio
import aiohttp
import json

DB_LOCK = threading.Lock()
PROGRESS_LOCK = threading.Lock()


# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
for logger_name in ["langchain", "langchain_community", "pymongo", "faiss", "urllib3", "requests"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# Global variables
GLOBAL_KNOWLEDGE_BASE = None
GLOBAL_QA_CHAIN = None
SCAN_PROGRESS = {"total": 0, "scanned": 0, "status": "not_started"}

# Configuration
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "lala"
COLLECTION_NAME = "update"
MAX_TOKENS = 2048
KB_PATH = None

def set_kb_path(kb_folder_path):
    """Set the knowledge base folder path."""
    global KB_PATH
    KB_PATH = kb_folder_path
    logging.info(f"Knowledge base path set to: {KB_PATH}")

def get_source_files(folder_path):
    """Get all source code files from the folder."""
    supported_extensions = ["py", "js", "java", "c", "cpp", "go", "php", "rb", "ts", "jsx", "html", "css"]
    files = []
    for ext in supported_extensions:
        pattern = os.path.join(folder_path, f"**/*.{ext}")
        files.extend(glob.glob(pattern, recursive=True))
    return files

def get_llm():
    """Load LLM model with optimized settings."""
    return Ollama(
        model="codellama:13b",
        temperature=0.1,
        num_ctx=4096,
        num_predict=1024
    )

def get_embeddings():
    """Load embeddings model."""
    return OllamaEmbeddings(model="codellama:13b")

def initialize_knowledge_base():
    """Initialize the knowledge base from FAISS and pickle files."""
    global GLOBAL_KNOWLEDGE_BASE, GLOBAL_QA_CHAIN
    
    if GLOBAL_KNOWLEDGE_BASE is not None:
        return GLOBAL_KNOWLEDGE_BASE
    
    if not KB_PATH:
        raise ValueError("Knowledge base path not set. Please call set_kb_path() first.")
    
    faiss_index_path = os.path.join(KB_PATH, "index.faiss")
    pkl_path = os.path.join(KB_PATH, "index.pkl")
    
    # Check if both files exist
    if not os.path.exists(faiss_index_path):
        raise FileNotFoundError(f"FAISS index file not found: {faiss_index_path}")
    
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Pickle file not found: {pkl_path}")
    
    logging.info(f"Loading FAISS index from: {faiss_index_path}")
    logging.info(f"Loading pickle file from: {pkl_path}")
    
    try:
        embeddings = get_embeddings()
        GLOBAL_KNOWLEDGE_BASE = FAISS.load_local(
            folder_path=KB_PATH,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Initialize QA chain once
        llm = get_llm()
        GLOBAL_QA_CHAIN = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=GLOBAL_KNOWLEDGE_BASE.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            ),
            verbose=False,
            return_source_documents=True
        )
        
        logging.info("Knowledge base and QA chain initialized successfully!")
        return GLOBAL_KNOWLEDGE_BASE
        
    except Exception as e:
        logging.exception("Error initializing knowledge base:")
        raise

def extract_vulnerable_function(file_content, snippet):
    """Extract the function or class surrounding the vulnerable code."""
    if not snippet or not snippet.strip():
        return "No code snippet available"
    
    try:
        lines = file_content.split('\n')
        snippet_lines = snippet.split('\n')
        
        # Find the line number where the snippet appears
        snippet_line = -1
        for i, line in enumerate(lines):
            if snippet_lines[0].strip() and snippet_lines[0].strip() in line:
                snippet_line = i
                break
        
        if snippet_line == -1:
            return snippet
        
        # Look backwards for function/class definition
        start_line = snippet_line
        function_keywords = [
            'def ', 'function ', 'class ', 'public ', 'private ', 'protected ',
            'static ', 'async ', 'func ', 'proc ', 'sub ', 'method '
        ]
        
        for i in range(snippet_line, max(-1, snippet_line - 50), -1):
            line = lines[i].strip()
            if any(keyword in line for keyword in function_keywords):
                start_line = i
                break
        
        # Look forward for reasonable end point
        end_line = min(snippet_line + 25, len(lines) - 1)
        
        # If we have a substantial context, return it
        extracted = '\n'.join(lines[start_line:end_line + 1])
        return extracted if len(extracted) > len(snippet) else snippet
        
    except Exception as e:
        logging.warning(f"Error extracting function context: {e}")
        return snippet

def clean_and_validate_field(field_value, max_length=None):
    """Clean and validate field values."""
    if not field_value:
        return ""
    
    # Remove extra whitespace and newlines
    cleaned = ' '.join(field_value.strip().split())
    
    # Apply length limit if specified
    if max_length and len(cleaned) > max_length:
        cleaned = cleaned[:max_length-3] + "..."
    
    return cleaned

def extract_relevant_info(output, file_name, engagement, owner, file_content):
    """Extract relevant security vulnerabilities from the LLM output."""
    if not output or not isinstance(output, str):
        return []

    vulnerabilities = []

    # Enhanced pattern matching for different output formats
    patterns = [
        re.compile(
            r"Vulnerability:\s*(.*?)\s*\n"
            r"CWE:\s*(.*?)\s*\n"
            r"Severity:\s*(.*?)\s*\n"
            r"Impact:\s*(.*?)\s*\n"
            r"Mitigation:\s*(.*?)\s*\n"
            r"Affected:\s*(.*?)\s*\n"
            r"Code Snippet:\s*(.*?)\s*(?=\n\n|\nVulnerability:|\Z)",
            re.DOTALL | re.IGNORECASE
        ),
        re.compile(
            r"(?:Issue|Problem|Security Issue):\s*(.*?)\n"
            r"(?:CWE|Type|Category):\s*(.*?)\n"
            r"(?:Risk|Severity|Level):\s*(.*?)\n"
            r"(?:Description|Impact|Risk Description):\s*(.*?)\n"
            r"(?:Fix|Solution|Mitigation|Recommendation):\s*(.*?)\n"
            r"(?:Location|File|Line|Function):\s*(.*?)\n"
            r"(?:Code|Snippet|Example):\s*(.*?)\s*(?=\n\n|\nIssue:|\Z)",
            re.DOTALL | re.IGNORECASE
        )
    ]

    for pattern in patterns:
        matches = pattern.findall(output)

        for match in matches:
            if len(match) < 7:
                continue

            try:
                title = clean_and_validate_field(match[0], 200)
                cwe = clean_and_validate_field(match[1], 100)
                severity = clean_and_validate_field(match[2], 20).upper()
                impact = clean_and_validate_field(match[3], 1000)
                mitigation = clean_and_validate_field(match[4], 1000)
                affected = clean_and_validate_field(match[5], 200)
                code_snippet = clean_and_validate_field(match[6], 500)

                if not title or not impact:
                    continue

                severity_mapping = {
                    'CRITICAL': ('CRITICAL', '9.8', 'CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H'),
                    'HIGH':     ('HIGH', '8.8', 'CVSS:3.1/AV:N/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H'),
                    'MEDIUM':   ('MEDIUM', '6.5', 'CVSS:3.1/AV:N/AC:L/PR:L/UI:N/S:U/C:L/I:L/A:L'),
                    'LOW':      ('LOW', '3.1', 'CVSS:3.1/AV:L/AC:H/PR:L/UI:R/S:U/C:L/I:N/A:N')
                }

                severity_normalized, cvss_score, cvss_vector = severity_mapping.get(severity, ('MEDIUM', '5.0', 'CVSS:3.1/AV:L/AC:H/PR:L/UI:N/S:U/C:L/I:L/A:N'))

                full_vulnerable_code = extract_vulnerable_function(file_content, code_snippet)

                if cwe and not cwe.upper().startswith('CWE-'):
                    if cwe.isdigit():
                        cwe = f"CWE-{cwe}"
                    elif not any(keyword in cwe.lower() for keyword in ['cwe', 'common weakness']):
                        cwe = f"CWE-{cwe}"

                cwe_number_match = re.search(r'CWE-(\d+)', cwe)
                cwe_number = cwe_number_match.group(1) if cwe_number_match else "NA"
                reference_link = f"https://cwe.mitre.org/data/definitions/{cwe_number}.html" if cwe_number != "NA" else "NA"

                # Auto-generate a PoC description
                pocdesc = f"This proof of concept demonstrates how {title.lower()} can occur. {impact}" if title and impact else "No PoC description provided."

                vulnerability = {
                    "title": title,
                    "cwe": cwe or "CWE-Unknown",
                    "severity": severity_normalized,
                    "security_risk": impact,
                    "mitigation": mitigation,
                    "affected_url": f"{file_name} - {affected}",
                    "engagement": ObjectId(engagement),
                    "code": 'F-' + str(uuid.uuid4()).replace('-', '')[:8],
                    "owner": ObjectId(owner),
                    "cvss_vector": cvss_vector,
                    "cvss_score": cvss_score,
                    "reference": reference_link,
                    "assistance": "ASCR",
                    "pocpic": full_vulnerable_code[:2000],
                    "pocdesc": pocdesc,
                    "rpoc": "[]",
                    "rpocdesc": "[]",
                    "under_reval": False,
                    "approved": False,
                    "status": "OPEN",
                    "deleted": False,
                    "createdat": datetime.utcnow().isoformat(timespec='milliseconds') + 'Z',
                }

                vulnerabilities.append(vulnerability)

            except Exception as e:
                logging.warning(f"Error processing vulnerability match: {e}")
                continue

    return vulnerabilities


def create_enhanced_prompt(chunk, file_name, file_extension):
    """Create an enhanced prompt tailored to the file type."""
    
    # File-specific vulnerability patterns
    vulnerability_focus = {
        'php': [
            'SQL Injection via unsanitized database queries',
            'Cross-Site Scripting (XSS) in output',
            'File inclusion vulnerabilities',
            'Authentication bypass',
            'Session management issues',
            'Command injection',
            'Path traversal'
        ],
        'js': [
            'Cross-Site Scripting (XSS)',
            'Prototype pollution',
            'Code injection via eval()',
            'DOM-based XSS',
            'Insecure API calls',
            'Client-side validation bypass'
        ],
        'py': [
            'SQL Injection',
            'Command injection',
            'Path traversal',
            'Insecure deserialization',
            'Code injection via exec/eval',
            'LDAP injection',
            'Template injection'
        ],
        'java': [
            'SQL Injection',
            'XML External Entity (XXE)',
            'Insecure deserialization',
            'Path traversal',
            'LDAP injection',
            'Expression Language injection'
        ],
        'c': [
            'Buffer overflow',
            'Use after free',
            'Format string vulnerabilities',
            'Integer overflow',
            'Null pointer dereference',
            'Race conditions'
        ],
        'cpp': [
            'Buffer overflow',
            'Use after free',
            'Memory corruption',
            'Integer overflow',
            'Double free',
            'Stack overflow'
        ]
    }
    
    focus_areas = vulnerability_focus.get(file_extension, [
        'Injection vulnerabilities',
        'Authentication issues',
        'Authorization bypass',
        'Input validation problems',
        'Output encoding issues'
    ])
    
    focus_text = '\n'.join([f"- {area}" for area in focus_areas])
    
    return f"""
You are a security expert analyzing {file_extension.upper()} code for vulnerabilities.

FOCUS AREAS for {file_extension.upper()}:
{focus_text}

IMPORTANT INSTRUCTIONS:
1. Only report ACTUAL security vulnerabilities, not code quality issues
2. Be specific about the vulnerability type and impact
3. Provide concrete mitigation steps
4. Use the exact format specified below

For each vulnerability found, use this EXACT format:

Vulnerability: [Specific vulnerability name]
CWE: [CWE-XXX format with description]
Severity: [Critical/High/Medium/Low]
Impact: [Detailed explanation of security impact and potential exploitation]
Mitigation: [Specific technical steps to fix the vulnerability]
Affected: [Function/method name and approximate line numbers]
Code Snippet: [The exact vulnerable code lines]

ANALYZE THIS CODE:
```{file_extension}
{chunk}
```

File: {file_name}

Remember: Only report actual security vulnerabilities with clear exploitation potential.
"""

def analyze_code_chunk(chunk, file_name, file_extension, engagement, owner, file_content):
    """Analyze a single code chunk for vulnerabilities."""
    if not GLOBAL_QA_CHAIN:
        raise RuntimeError("QA chain not initialized. Call initialize_knowledge_base() first.")
    
    findings = []
    
    try:
        prompt = create_enhanced_prompt(chunk, file_name, file_extension)
        
        # Query the knowledge base
        result = GLOBAL_QA_CHAIN.invoke({"query": prompt})

        
        if result and "result" in result:
            output = result["result"]
            findings = extract_relevant_info(output, file_name, engagement, owner, file_content)
            
            # Log source documents for debugging
            if "source_documents" in result and result["source_documents"]:
                logging.debug(f"Used {len(result['source_documents'])} knowledge base documents")
        
    except Exception as e:
        logging.error(f"Error analyzing chunk: {str(e)}")
    
    return findings

def scan_single_file(file_path, engagement, owner, max_chunk_batch=5, cache_enabled=True):
    """Scan a single file for security vulnerabilities with advanced optimizations."""
    file_name = os.path.basename(file_path)
    file_extension = os.path.splitext(file_name)[1][1:].lower()
    
    logging.info(f"Scanning {file_name}...")
    
    # Simple in-memory cache for retrieval results
    retrieval_cache = {}
    
    def cache_key(prompt):
        return hashlib.sha256(prompt.encode()).hexdigest()
    
    async def async_invoke(session, prompt):
        # Synchronous fallback for Ollama since async might not be supported
        try:
            # Use the existing QA chain for consistency
            result = GLOBAL_QA_CHAIN.invoke({"query": prompt})
            if result and "result" in result:
                return result["result"]
            return ""
        except Exception as e:
            logging.error(f"Error in async invoke: {e}")
            return ""
    
    def batch_chunks(chunks, batch_size):
        for i in range(0, len(chunks), batch_size):
            yield chunks[i:i + batch_size]
    
    async def analyze_chunk_batch_async(session, chunk_batch):
        all_findings = []
        for chunk in chunk_batch:
            prompt = create_enhanced_prompt(chunk, file_name, file_extension)
            key = cache_key(prompt)
            
            if cache_enabled and key in retrieval_cache:
                logging.debug(f"Cache hit for chunk in {file_name}")
                output = retrieval_cache[key]
            else:
                # Check system resource usage to avoid overload
                cpu_percent = psutil.cpu_percent(interval=0.1)
                mem = psutil.virtual_memory()
                if cpu_percent > 90 or mem.percent > 90:
                    logging.warning(f"High system load detected (CPU: {cpu_percent}%, Memory: {mem.percent}%), delaying chunk analysis")
                    await asyncio.sleep(1)
                
                output = await async_invoke(session, prompt)
                if cache_enabled:
                    retrieval_cache[key] = output
            
            findings = extract_relevant_info(output, file_name, engagement, owner, file_content)
            all_findings.extend(findings)
        
        return all_findings
    
    async def process_chunks_async(chunks):
        all_findings = []
        seen_hashes = set()
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for chunk_batch in batch_chunks(chunks, max_chunk_batch):
                tasks.append(analyze_chunk_batch_async(session, chunk_batch))
            
            results = await asyncio.gather(*tasks)
            for findings in results:
                for finding in findings:
                    key_str = finding.get("title", "") + finding.get("cwe", "") + finding.get("affected_url", "")
                    key_hash = hashlib.sha256(key_str.encode()).hexdigest()
                    if key_hash not in seen_hashes:
                        seen_hashes.add(key_hash)
                        all_findings.append(finding)
        
        return all_findings
    
    try:
        # Read file content
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            file_content = f.read()
        
        # Skip empty or very small files
        if len(file_content.strip()) < 50:
            logging.info(f"Skipping {file_name} - file too small or empty")
            return []
        
        # Create overlapping chunks for better context, ensuring full coverage without gaps
        chunk_size = MAX_TOKENS
        overlap = 300
        chunks = []
        start = 0
        while start < len(file_content):
            end = min(start + chunk_size, len(file_content))
            chunk = file_content[start:end]
            if chunk.strip():
                chunks.append(chunk)
            if end == len(file_content):
                break
            start = end - overlap  # overlap to avoid missing context
        
        # If file is small, use as single chunk
        if len(chunks) == 0:
            chunks = [file_content]
        
        # Run async processing of chunk batches
        all_findings = asyncio.run(process_chunks_async(chunks))
        
        logging.info(f"Found {len(all_findings)} unique vulnerabilities in {file_name}")
        return all_findings
        
    except Exception as e:
        logging.error(f"Error scanning {file_path}: {str(e)}")
        return []

def save_findings_to_db(findings):
    """Save findings to MongoDB database."""
    if not findings:
        return
    
    with DB_LOCK:
        client = MongoClient(MONGO_URI)
        try:
            db = client[DB_NAME]
            collection = db[COLLECTION_NAME]
            result = collection.insert_many(findings)
            logging.info(f"Saved {len(result.inserted_ids)} findings to database")
        except Exception as e:
            logging.error(f"Error saving findings to database: {e}")
        finally:
            client.close()


def update_progress(current, total):
    """Update scan progress."""
    with PROGRESS_LOCK:
        SCAN_PROGRESS.update({
            "scanned": current,
            "total": total,
            "percentage": round((current / total) * 100, 2) if total > 0 else 0
        })
        # Update severity counts during scan progress if project_name is available
        project_name = SCAN_PROGRESS.get("project_name", "")
        if project_name:
            SCAN_PROGRESS["severity_counts"] = get_severity_counts(project_name)


def scan_folder(folder_path, engagement, owner, kb_path, project_name=None):
    """
    Scan a folder for security vulnerabilities.
    
    Args:
        folder_path (str): Path to the folder containing source code files
        engagement (str): Engagement ID
        owner (str): Owner ID
        kb_path (str): Path to the knowledge base
        project_name (str, optional): Name of the project
    
    Returns:
        list: List of all findings
    """
    global SCAN_PROGRESS

    # Set knowledge base path and initialize
    set_kb_path(kb_path)
    initialize_knowledge_base()

    # Get all source files
    source_files = get_source_files(folder_path)
    total_files = len(source_files)

    if not source_files:
        logging.warning(f"No source code files found in {folder_path}")
        return []

    # Initialize progress tracking
    SCAN_PROGRESS = {
        "total": total_files,
        "scanned": 0,
        "status": "running",
        "start_time": datetime.now().isoformat(),
        "percentage": 0,
        "project_name": project_name or "Unknown Project"
    }

    logging.info(f"Starting parallel scan of {total_files} files for project: {project_name or 'Unknown'}")

    total_vulns = 0
    all_findings = []

    # Use thread pool for parallel processing
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {
            executor.submit(scan_single_file, file_path, engagement, owner): file_path
            for file_path in source_files
        }

        for index, future in enumerate(as_completed(futures)):
            file_path = futures[future]
            try:
                findings = future.result()
                total_vulns += len(findings)

                if findings:
                    save_findings_to_db(findings)
                    all_findings.extend(findings)

            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}\n{traceback.format_exc()}")

            finally:
                update_progress(index + 1, total_files)
                display_progress()

    # Mark scan as completed
    SCAN_PROGRESS.update({
        "status": "completed",
        "end_time": datetime.now().isoformat()
    })

    logging.info(f"Scan completed! Found {total_vulns} total vulnerabilities across {total_files} files")
    return all_findings

def get_scan_progress():
    """Get current scan progress."""
    return SCAN_PROGRESS.copy()

def get_severity_counts(engagement_id):
    """Fetch severity counts from MongoDB for the given engagement ID."""
    from pymongo import MongoClient
    client = MongoClient(MONGO_URI)
    try:
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        pipeline = [
            {"$match": {"engagement": ObjectId(engagement_id)}},
            {"$group": {"_id": "$severity", "count": {"$sum": 1}}}
        ]
        results = collection.aggregate(pipeline)
        counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for doc in results:
            severity = doc["_id"].upper()
            if severity in counts:
                counts[severity] = doc["count"]
        return counts
    except Exception as e:
        logging.error(f"Error fetching severity counts: {e}")
        return {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    finally:
        client.close()

def scan_folder(folder_path, engagement, owner, kb_path, project_name=None):
    """
    Scan a folder for security vulnerabilities.
    
    Args:
        folder_path (str): Path to the folder containing source code files
        engagement (str): Engagement ID
        owner (str): Owner ID
        kb_path (str): Path to the knowledge base
        project_name (str, optional): Name of the project
    
    Returns:
        list: List of all findings
    """
    global SCAN_PROGRESS

    # Set knowledge base path and initialize
    set_kb_path(kb_path)
    initialize_knowledge_base()

    # Get all source files
    source_files = get_source_files(folder_path)
    total_files = len(source_files)

    if not source_files:
        logging.warning(f"No source code files found in {folder_path}")
        return []

    # Initialize progress tracking
    SCAN_PROGRESS = {
        "total": total_files,
        "scanned": 0,
        "status": "running",
        "start_time": datetime.now().isoformat(),
        "percentage": 0,
        "project_name": project_name or "Unknown Project",
        "engagement": engagement,
        "severity_counts": {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    }

    logging.info(f"Starting parallel scan of {total_files} files for project: {project_name or 'Unknown'}")

    total_vulns = 0
    all_findings = []

    # Use thread pool for parallel processing
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {
            executor.submit(scan_single_file, file_path, engagement, owner): file_path
            for file_path in source_files
        }

        for index, future in enumerate(as_completed(futures)):
            file_path = futures[future]
            try:
                findings = future.result()
                total_vulns += len(findings)

                if findings:
                    save_findings_to_db(findings)
                    all_findings.extend(findings)

            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}\n{traceback.format_exc()}")

            finally:
                update_progress(index + 1, total_files)
                display_progress()

    # After scan completion, update severity counts from DB
    SCAN_PROGRESS["severity_counts"] = get_severity_counts(SCAN_PROGRESS.get("engagement", ""))

    # Mark scan as completed
    SCAN_PROGRESS.update({
        "status": "completed",
        "end_time": datetime.now().isoformat()
    })

    logging.info(f"Scan completed! Found {total_vulns} total vulnerabilities across {total_files} files")
    return all_findings

def cleanup_resources():
    """Clean up global resources."""
    global GLOBAL_KNOWLEDGE_BASE, GLOBAL_QA_CHAIN
    GLOBAL_KNOWLEDGE_BASE = None
    GLOBAL_QA_CHAIN = None
    logging.info("Resources cleaned up")
    
    
def display_progress():
    """Print a formatted view of scan progress."""
    scanned = SCAN_PROGRESS.get("scanned", 0)
    total = SCAN_PROGRESS.get("total", 0)
    remaining = total - scanned
    percentage = SCAN_PROGRESS.get("percentage", 0)
    status = SCAN_PROGRESS.get("status", "unknown")
    start_time = SCAN_PROGRESS.get("start_time", "N/A")
    end_time = SCAN_PROGRESS.get("end_time", "N/A")
    project_name = SCAN_PROGRESS.get("project_name", "Unknown Project")

    print("\n" + "=" * 50)
    print("SCAN PROGRESS")
    print("=" * 50)
    print(f"Project          : {project_name}")
    print(f"Status           : {status.upper()}")
    print(f"Start Time       : {start_time}")
    if status == "completed":
        print(f"End Time         : {end_time}")
    print(f"Total Files      : {total}")
    print(f"Files Scanned    : {scanned}")
    print(f"Files Remaining  : {remaining}")
    print(f"Completed        : {percentage}%")
    print("=" * 50 + "\n")


# This module is designed to be imported and used by Django views
# All scanning operations are handled through the Django API endpoints