from pydantic import BaseModel, Field
from typing import List, Optional

# --- PHASE 2: Mapper Schemas ---

class ModuleSummary(BaseModel):
    file_path: str
    purpose: str = Field(description="Max 1 short sentence describing the file's main responsibility")
    key_functions: List[str] = Field(description="List of core function names only (no descriptions)")
    dependant_count: int = 0

# --- PHASE 1.5: Triage Schemas ---

class TriageDecision(BaseModel):
    complexity_score: int = Field(description="1 to 10 scale of change complexity")
    is_trivial: bool = Field(description="True if only docs, comments, or formatting changed (no logic)")
    needs_security_review: bool = Field(description="True if touching auth, secrets, inputs, db queries, core config")
    needs_qa_review: bool = Field(description="True if touching core logic, state, edge cases, or adding major features")

# --- PHASE 3: Analysis Chain Schemas ---

class Finding(BaseModel):
    severity: str = Field(description="Must be: CRITICAL, HIGH, MEDIUM, LOW, or INFO")
    category: str = Field(description="Type of issue, max 3 words")
    title: str = Field(description="Wait, short 3-7 word summary of the issue")
    line_number: Optional[int] = Field(default=None, description="The exact integer line number where the issue occurs. You must provide an integer (guess based on the snippet if exact line is unknown).")
    class_or_method: Optional[str] = Field(default=None, description="The class/method name where the issue resides. Use 'Global' if it is a module-level issue.")
    description: str = Field(description="Explanation of what is wrong (MAX 2 SHORT SENTENCES)")
    suggestion: str = Field(description="Actionable fix or code snippet (MAX 2 SHORT SENTENCES)")
    confidence: float = Field(description="Score between 0.0 and 1.0")

class FindingList(BaseModel):
    findings: List[Finding]

class FileFinding(BaseModel):
    file_path: str = Field(description="Relative path of the file")
    purpose: str = Field(default="", description="Max 1 short sentence describing the file's main core responsibility")
    key_functions: List[str] = Field(default_factory=list, description="List of core function names only (no descriptions). Must be EXACT names.")
    findings: List[Finding] = Field(description="Security findings for this file. Leave empty list if none.")

class BatchFindingList(BaseModel):
    files: List[FileFinding] = Field(description="Findings grouped by file")

# --- PHASE 4: BRD Verifier Schemas ---

class Requirement(BaseModel):
    id: int = Field(description="A unique incremental integer ID")
    description: str = Field(description="The atomic business requirement extracted from the text")
    category: str = Field(description="e.g., Security, Logic, UI, Database, Performance")

class RequirementList(BaseModel):
    requirements: List[Requirement]

class VerificationResult(BaseModel):
    requirement_id: int
    status: str = Field(description="Must be exactly: MET, NOT_MET, or PARTIAL")
    justification: str = Field(description="Max 1 short sentence justifying the status")
    evidence_files: List[str] = Field(description="List of up to 3 relevant file paths")

class VerificationResultList(BaseModel):
    results: List[VerificationResult]