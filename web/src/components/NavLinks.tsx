"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Console" },
  { href: "/about", label: "How to read this" },
];

/** Header navigation; the current page is underlined and announced. */
export function NavLinks() {
  const pathname = usePathname();
  return (
    <nav aria-label="Main" className="flex gap-1 sm:ml-auto">
      {LINKS.map(({ href, label }) => {
        const current = pathname === href;
        return (
          <Link
            key={href}
            href={href}
            aria-current={current ? "page" : undefined}
            className={`border-b-4 px-3 pb-2 pt-2.5 text-[16px] font-bold transition-colors ${
              current
                ? "border-white text-white"
                : "border-transparent text-white/85 hover:border-white/50 hover:text-white"
            }`}
          >
            {label}
          </Link>
        );
      })}
    </nav>
  );
}
