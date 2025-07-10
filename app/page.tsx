"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Drawer, DrawerContent, DrawerTrigger } from "@/components/ui/drawer";
import {
  Bot,
  CircleAlert,
  CircleAlertIcon,
  DoorClosedIcon,
  FileCheck2,
  LucideCircleAlert,
  OctagonAlert,
  Plus,
  Settings,
  TriangleAlert,
  ChevronsDown,
  Github,
  Menu,
} from "lucide-react";
import { ModeToggle } from "@/components/modetoggle";
import { useChat } from "ai/react";
import ReportComponent from "@/components/ReportComponent";
import { useToast } from "@/components/ui/use-toast";
import ChatComponent from "@/components/chatcomponent";
import Link from "next/link";
import { Sheet, SheetContent, SheetFooter, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { Separator } from "@/components/ui/separator";

const routeList = [
  { href: "/", label: "Home" },
  { href: "/searchmedicine", label: "Medicine GPT" }
];

const Home = () => {
  const { toast } = useToast();
  const [reportData, setreportData] = useState("");
  const [isOpen, setIsOpen] = useState(false);

  const onReportConfirmation = (data: string) => {
    setreportData(data);
    toast({
      description: "Updated!"
    });
  };

  return (
    <div className="grid h-screen w-full">
      <div className="flex flex-col">
        {/* New Navbar */}
        <header className="shadow-inner bg-opacity-15 w-[90%] md:w-[70%] lg:w-[75%] lg:max-w-screen-xl top-5 mx-auto sticky border border-secondary z-40 rounded-2xl flex justify-between items-center p-2 bg-card mb-10">
          <Link href="/" className="font-bold text-lg flex items-center">
            <ChevronsDown className="bg-gradient-to-tr border-secondary from-primary via-primary/70 to-primary rounded-lg w-9 h-9 mr-2 border text-white" />
            RAG Powered Medical Assistant
          </Link>
          
          {/* Mobile Menu */}
          <div className="flex items-center lg:hidden">
            <Sheet open={isOpen} onOpenChange={setIsOpen}>
              <SheetTrigger asChild>
                <Menu onClick={() => setIsOpen(!isOpen)} className="cursor-pointer lg:hidden" />
              </SheetTrigger>
              <SheetContent
                side="left"
                className="flex flex-col justify-between rounded-tr-2xl rounded-br-2xl bg-card border-secondary"
              >
                <div>
                  <SheetHeader className="mb-4 ml-4">
                    <SheetTitle className="flex items-center">
                      <Link href="/" className="flex items-center">
                        <ChevronsDown className="bg-gradient-to-tr border-secondary from-primary via-primary/70 to-primary rounded-lg w-9 h-9 mr-2 border text-white" />
                        MedNourish AI
                      </Link>
                    </SheetTitle>
                  </SheetHeader>
                  <div className="flex flex-col gap-2">
                    {routeList.map(({ href, label }) => (
                      <Button
                        key={href}
                        onClick={() => setIsOpen(false)}
                        asChild
                        variant="ghost"
                        className="justify-start text-base"
                      >
                        <Link href={href}>{label}</Link>
                      </Button>
                    ))}
                  </div>
                </div>
                <SheetFooter className="flex-col sm:flex-col justify-start items-start">
                  <Separator className="mb-2" />
                </SheetFooter>
              </SheetContent>
            </Sheet>
          </div>

          {/* Desktop Menu */}
          <div className="hidden lg:flex items-center">
            {routeList.map(({ href, label }) => (
              <Button key={href} asChild variant="ghost" className="text-base px-2">
                <Link href={href}>{label}</Link>
              </Button>
            ))}
            <div className="flex items-center ml-4">
              <Button asChild size="sm" variant="ghost" aria-label="View on GitHub">
                <Link
                  aria-label="View on GitHub"
                  href="https://github.com"
                  target="_blank"
                >
                  <Github className="size-5" />
                </Link>
              </Button>
              <ModeToggle />
              <Drawer>
                <DrawerTrigger asChild>
                  <Button variant="ghost" size="icon" className="md:hidden">
                    <Settings />
                    <span className="sr-only">Settings</span>
                  </Button>
                </DrawerTrigger>
                <DrawerContent className="max-h-[80vh]">
                  <ReportComponent onReportConfirmation={onReportConfirmation} />
                </DrawerContent>
              </Drawer>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="grid flex-1 gap-4 overflow-auto p-4 md:grid-cols-2 lg:grid-cols-3">
          <div className="hidden md:flex flex-col">
            <ReportComponent onReportConfirmation={onReportConfirmation} />
          </div>
          <div className="lg:col-span-2">
            <ChatComponent reportData={reportData} />
          </div>
        </main>
      </div>
    </div>
  );
};

export default Home;