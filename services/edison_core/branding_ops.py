"""
Branding and marketing workflow generation with project/client output storage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import UTC, datetime
import json
import re

from .branding_store import BrandingClientStore
from .contracts import BrandingGenerationRequest, MarketingCopyRequest
from .projects import ProjectWorkspaceManager


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (value or "").strip().lower()).strip("-") or "item"


class BrandingWorkflowService:
    def __init__(
        self,
        repo_root: Path,
        branding_store: BrandingClientStore,
        project_manager: ProjectWorkspaceManager,
    ):
        self.repo_root = repo_root.resolve()
        self.branding_store = branding_store
        self.project_manager = project_manager

    def generate_brand_package(self, request: BrandingGenerationRequest) -> Dict[str, Any]:
        workspace_dir, context = self._resolve_workspace(request.client_id, request.project_id, "branding")
        package = self._build_brand_package(request, context)
        stamp = _utc_stamp()
        outputs = [
            self._write_json(workspace_dir / f"logo_concepts_{stamp}.json", package["logo_concepts"], "logo_concepts", request.project_id),
            self._write_markdown(workspace_dir / f"moodboard_{stamp}.md", package["moodboard"], "moodboard", request.project_id),
            self._write_json(workspace_dir / f"palette_{stamp}.json", package["palette"], "palette", request.project_id),
            self._write_json(workspace_dir / f"typography_{stamp}.json", package["typography"], "typography", request.project_id),
            self._write_json(workspace_dir / f"slogans_{stamp}.json", package["slogans"], "slogans", request.project_id),
            self._write_markdown(workspace_dir / f"brand_voice_{stamp}.md", package["brand_voice"], "brand_voice", request.project_id),
            self._write_markdown(workspace_dir / f"style_guide_{stamp}.md", package["style_guide"], "style_guide", request.project_id),
        ]
        return {
            "business_name": request.business_name,
            "project_id": request.project_id,
            "client_id": request.client_id,
            "outputs": outputs,
            "package": package,
            "workspace": str(workspace_dir),
        }

    def generate_marketing_copy(self, request: MarketingCopyRequest) -> Dict[str, Any]:
        workspace_dir, context = self._resolve_workspace(request.client_id, request.project_id, "marketing")
        copy_bundle = self._build_marketing_copy(request, context)
        stamp = _utc_stamp()
        outputs = []
        for kind, payload in copy_bundle.items():
            outputs.append(self._write_json(workspace_dir / f"{kind}_{stamp}.json", payload, kind, request.project_id))
        return {
            "business_name": request.business_name,
            "project_id": request.project_id,
            "client_id": request.client_id,
            "outputs": outputs,
            "copy": copy_bundle,
            "workspace": str(workspace_dir),
        }

    def _resolve_workspace(self, client_id: Optional[str], project_id: Optional[str], category: str) -> Tuple[Path, Dict[str, Any]]:
        context: Dict[str, Any] = {}
        if project_id:
            project = self.project_manager.get_project(project_id)
            if not project:
                raise ValueError("Project not found")
            context["project"] = project
            workspace_dir = Path(project.workspace_paths.get(category) or project.root_path).resolve(strict=False)
            workspace_dir.mkdir(parents=True, exist_ok=True)
            return workspace_dir, context
        if client_id:
            client = self.branding_store.get_client(client_id)
            if not client:
                raise ValueError("Client not found")
            context["client"] = client
            workspace_dir = self.branding_store.get_generation_dir(client_id, category)
            workspace_dir.mkdir(parents=True, exist_ok=True)
            return workspace_dir, context
        fallback = (self.repo_root / "outputs" / category / _slugify(_utc_stamp())).resolve(strict=False)
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback, context

    def _write_json(self, path: Path, payload: Any, title: str, project_id: Optional[str]) -> Dict[str, Any]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
        return self._register_output(path, title, project_id, "json")

    def _write_markdown(self, path: Path, content: str, title: str, project_id: Optional[str]) -> Dict[str, Any]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return self._register_output(path, title, project_id, "markdown")

    def _register_output(self, path: Path, title: str, project_id: Optional[str], content_type: str) -> Dict[str, Any]:
        rel = self._relative(path)
        if project_id:
            self.project_manager.register_output(
                project_id=project_id,
                path=rel,
                category=title,
                title=title.replace("_", " ").title(),
                metadata={"content_type": content_type},
            )
        return {
            "title": title,
            "path": rel,
            "content_type": content_type,
        }

    def _relative(self, path: Path) -> str:
        try:
            return str(path.resolve(strict=False).relative_to(self.repo_root))
        except Exception:
            return str(path.resolve(strict=False))

    def _build_brand_package(self, request: BrandingGenerationRequest, context: Dict[str, Any]) -> Dict[str, Any]:
        business_name = request.business_name.strip()
        industry = request.industry.strip() or self._context_value(context, "industry") or "general business"
        audience = request.audience.strip() or "local customers seeking a memorable brand"
        style_keywords = request.style_keywords or self._infer_style_keywords(industry, request.prompt)
        tone = request.tone.strip() or "confident"
        core_words = self._core_words(business_name, style_keywords)

        logo_concepts = []
        for idx in range(request.deliverable_count):
            concept_word = core_words[idx % len(core_words)]
            logo_concepts.append({
                "concept": idx + 1,
                "name": f"{business_name} {concept_word.title()}",
                "direction": self._concept_direction(idx, style_keywords, industry),
                "shape_language": self._shape_language(idx),
                "applications": self._applications(industry),
            })

        palette = {
            "primary": self._colorway(style_keywords, 0),
            "secondary": self._colorway(style_keywords, 1),
            "accent": self._colorway(style_keywords, 2),
            "neutral": self._colorway(style_keywords, 3),
            "rationale": f"Palette balances a {tone} tone with cues from {industry} and the keywords {', '.join(style_keywords[:4])}.",
        }

        typography = {
            "headline": self._font_pick(style_keywords, "headline"),
            "subhead": self._font_pick(style_keywords, "subhead"),
            "body": self._font_pick(style_keywords, "body"),
            "usage_notes": f"Use tighter tracking for display text and keep body copy direct for {audience}.",
        }

        slogans = [
            f"{business_name}: {self._verb_for_industry(industry).title()} with purpose.",
            f"{business_name} makes {industry} feel unmistakably alive.",
            f"Bold {industry}. Clear promise. {business_name}.",
            f"Where {industry} meets {style_keywords[0] if style_keywords else 'clarity'}.",
            f"{business_name} for customers who want more than ordinary.",
        ][: request.deliverable_count]

        moodboard = "\n".join([
            f"# {business_name} Moodboard",
            "",
            f"- Industry: {industry}",
            f"- Audience: {audience}",
            f"- Tone: {tone}",
            f"- Visual anchors: {', '.join(style_keywords)}",
            "- Photography direction: tight detail shots, tactile materials, environmental context.",
            "- Layout direction: asymmetric editorial grids with strong hero moments and generous negative space.",
            f"- Texture ideas: {self._texture_notes(style_keywords)}",
        ])

        brand_voice = "\n".join([
            f"# {business_name} Brand Voice",
            "",
            f"{business_name} should sound {tone}, grounded, and useful.",
            f"Write to {audience} with short sentences, clear offers, and language that reflects {industry} expertise.",
            "Avoid filler, empty hype, and generic claims.",
            "Lead with outcomes, then reinforce craftsmanship, speed, or trust depending on context.",
        ])

        style_guide = "\n".join([
            f"# {business_name} Style Guide Starter",
            "",
            "## Brand Position",
            f"{business_name} occupies a {tone} position in {industry}, focused on {audience}.",
            "",
            "## Visual System",
            f"Prioritize {', '.join(style_keywords[:3])} across identity, packaging, social assets, and signage.",
            "",
            "## Messaging",
            f"Use the slogan line '{slogans[0]}' as the shortest expression of the offer.",
            "",
            "## Production Notes",
            "Keep vector marks simple enough for signage, embroidery, print, and 3D extrusion when needed.",
        ])

        return {
            "logo_concepts": logo_concepts,
            "moodboard": moodboard,
            "palette": palette,
            "typography": typography,
            "slogans": slogans,
            "brand_voice": brand_voice,
            "style_guide": style_guide,
        }

    def _build_marketing_copy(self, request: MarketingCopyRequest, context: Dict[str, Any]) -> Dict[str, Any]:
        business_name = request.business_name.strip()
        industry = request.industry.strip() or self._context_value(context, "industry") or "general business"
        audience = request.audience.strip() or "customers ready to buy"
        tone = request.tone.strip() or "confident"
        prompt = request.prompt.strip() or f"promote {business_name}"
        copy_types = request.copy_types or [
            "ad_copy",
            "social_captions",
            "business_description",
            "website_hero_text",
        ]
        bundle: Dict[str, Any] = {}
        if "ad_copy" in copy_types:
            bundle["ad_copy"] = {
                "headline": f"{business_name} brings a sharper standard to {industry}.",
                "body": f"For {audience}, {business_name} delivers {prompt} with a {tone} voice and a concrete next step.",
                "cta": f"Book {business_name} today.",
            }
        if "social_captions" in copy_types:
            bundle["social_captions"] = {
                "captions": [
                    f"{business_name} is building a more memorable {industry} experience. {prompt.capitalize()}. #branding #marketing",
                    f"Clear message. Strong visuals. {business_name} is ready for the next campaign.",
                    f"If your audience wants proof, show them how {business_name} turns {industry} into something worth sharing.",
                ],
                "channels": request.channels or ["instagram", "facebook", "linkedin"],
            }
        if "email_campaign" in copy_types:
            bundle["email_campaign"] = {
                "subject": f"See what {business_name} can do next",
                "preview": f"A sharper {industry} story, ready to launch.",
                "body": f"{business_name} is rolling out a {tone} campaign focused on {audience}. {prompt.capitalize()} with a direct offer and a simple reply path.",
            }
        if "business_description" in copy_types:
            bundle["business_description"] = {
                "short": f"{business_name} is a {industry} brand focused on {audience}.",
                "long": f"{business_name} combines clear positioning, distinctive visuals, and practical execution to help {audience} choose with confidence.",
            }
        if "product_copy" in copy_types:
            bundle["product_copy"] = {
                "headline": f"Built for {audience}",
                "body": f"This offer from {business_name} translates {industry} expertise into something tangible, immediate, and easy to trust.",
            }
        if "website_hero_text" in copy_types:
            bundle["website_hero_text"] = {
                "headline": f"{business_name} makes {industry} feel decisive.",
                "subheadline": f"A {tone} brand system built for {audience}.",
                "cta_primary": "Start the project",
                "cta_secondary": "See the work",
            }
        return bundle

    def _context_value(self, context: Dict[str, Any], key: str) -> str:
        client = context.get("client")
        project = context.get("project")
        if client and getattr(client, key, None):
            return getattr(client, key)
        if isinstance(client, dict) and client.get(key):
            return str(client.get(key))
        if project and getattr(project, key, None):
            return getattr(project, key)
        return ""

    def _infer_style_keywords(self, industry: str, prompt: str) -> List[str]:
        words = [industry, prompt]
        blob = " ".join(words).lower()
        defaults = ["bold", "tactile", "editorial", "clear"]
        if any(word in blob for word in ["pizza", "restaurant", "food"]):
            return ["warm", "appetite", "street", "craft"]
        if any(word in blob for word in ["fabrication", "printing", "industrial"]):
            return ["precise", "modular", "durable", "technical"]
        if any(word in blob for word in ["luxury", "premium"]):
            return ["refined", "quiet", "minimal", "tailored"]
        return defaults

    def _core_words(self, business_name: str, style_keywords: List[str]) -> List[str]:
        tokens = [token for token in re.findall(r"[A-Za-z]+", business_name) if len(token) > 2]
        words = tokens + style_keywords + ["signal", "studio", "mark"]
        deduped = []
        seen = set()
        for word in words:
            lowered = word.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            deduped.append(word)
        return deduped[:8] or ["mark"]

    def _concept_direction(self, idx: int, style_keywords: List[str], industry: str) -> str:
        directions = [
            f"Monogram-led identity with {style_keywords[0] if style_keywords else 'bold'} emphasis and simplified geometry for {industry}.",
            f"Badge system using layered framing and tactile cues for merchandise, menus, and signage.",
            f"Wordmark-first direction with compact spacing, optimized for print, social, and extrusion.",
            f"Symbol plus wordmark pairing built around movement, rhythm, and instant recognition.",
            f"Heritage-inspired mark with modern simplification and high-contrast production behavior.",
        ]
        return directions[idx % len(directions)]

    def _shape_language(self, idx: int) -> str:
        values = ["rounded forms", "angular cuts", "stacked blocks", "single-stroke curves", "offset frames"]
        return values[idx % len(values)]

    def _applications(self, industry: str) -> List[str]:
        return [
            f"Primary storefront and {industry} signage",
            "Social templates and ad creatives",
            "Business cards, flyers, and apparel",
            "3D printable promotional items",
        ]

    def _colorway(self, keywords: List[str], idx: int) -> Dict[str, str]:
        palette = [
            {"name": "Brick Ember", "hex": "#B24C2E"},
            {"name": "Night Ink", "hex": "#1E293B"},
            {"name": "Signal Gold", "hex": "#D9A441"},
            {"name": "Warm Sand", "hex": "#E8D9C5"},
            {"name": "Ocean Slate", "hex": "#355C7D"},
            {"name": "Forest Alloy", "hex": "#476A53"},
        ]
        if any(word in (" ".join(keywords)).lower() for word in ["warm", "food", "craft"]):
            palette = [palette[0], palette[2], palette[3], palette[1]]
        return palette[idx % len(palette)]

    def _font_pick(self, keywords: List[str], slot: str) -> Dict[str, str]:
        families = {
            "headline": ["Space Grotesk", "Archivo Expanded", "Bricolage Grotesque"],
            "subhead": ["Sora", "General Sans", "Outfit"],
            "body": ["Source Serif 4", "IBM Plex Sans", "Fraunces"],
        }
        picks = families.get(slot, families["body"])
        tone = "high-contrast" if "refined" in keywords else "confident"
        return {"family": picks[0], "tone": tone}

    def _verb_for_industry(self, industry: str) -> str:
        lowered = industry.lower()
        if any(word in lowered for word in ["restaurant", "food", "pizza"]):
            return "serve"
        if any(word in lowered for word in ["fabrication", "printing", "manufacturing"]):
            return "build"
        return "deliver"

    def _texture_notes(self, keywords: List[str]) -> str:
        if "craft" in keywords:
            return "paper grain, worn metal, hand-painted type"
        if "technical" in keywords:
            return "anodized metal, cut acrylic, matte powder coat"
        return "soft shadows, layered paper, brushed surfaces"